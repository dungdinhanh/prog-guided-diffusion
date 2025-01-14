"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""

import argparse
import os

import numpy as np
import torch
import torch as th
# import torch.distributed as dist
import hfai.nccl.distributed as dist
import torch.nn.functional as F
import hfai
from guided_diffusion import dist_util, logger
from eds_guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)

from eds_guided_diffusion.script_util_robust import create_model_and_diffusion_lsim_sch_adaptwconf

import datetime
from PIL import Image
from torchvision import utils
import hfai.client
import hfai.multiprocessing


def main(local_rank):
    args = create_argparser().parse_args()

    dist_util.setup_dist(local_rank)

    if args.fix_seed:
        import random
        seed = 23333 + dist.get_rank()
        np.random.seed(seed)
        th.manual_seed(seed)  # CPU随机种子确定
        th.cuda.manual_seed(seed)  # GPU随机种子确定
        th.cuda.manual_seed_all(seed)  # 所有的GPU设置种子

        th.backends.cudnn.benchmark = False  # 模型卷积层预先优化关闭
        th.backends.cudnn.deterministic = True  # 确定为默认卷积算法

        random.seed(seed)
        np.random.seed(seed)

        os.environ['PYTHONHASHSEED'] = str(seed)

    save_folder = os.path.join(
        args.logdir,
        "logs",
    )

    logger.configure(save_folder, rank=dist.get_rank())

    output_images_folder = os.path.join(args.logdir, "reference")
    os.makedirs(output_images_folder, exist_ok=True)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion_lsim_sch_adaptwconf(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    diffusion.epsilon = args.eps

    logger.log("loading classifier...")
    classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
    classifier.load_state_dict(
        dist_util.load_state_dict(args.classifier_path, map_location="cpu")
    )
    classifier.to(dist_util.dev())
    if args.classifier_use_fp16:
        classifier.convert_to_fp16()
    classifier.eval()

    def cond_fn(x, t, y=None, cls=None):
        assert y is not None
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            probs = F.softmax(logits, dim=-1)
            entropy_logits = (-log_probs * probs).sum(dim=-1)
            entropy_labels = entropy_calculate(y)
            adaptive_delta_eps = torch.unsqueeze((entropy_labels/entropy_logits) * (1 - args.eps), dim=1)
            new_y = label_augment(y, cls, dist_util.dev(), adaptive_delta_eps)
            # selected = log_probs[range(len(logits)), y.view(-1)]

            loss = (new_y * log_probs).sum(-1)
            # return th.autograd.grad(selected.sum(), x_in)[0] * args.classifier_scale
            cond_grad = th.autograd.grad(loss.sum(), x_in)[0]

        guidance = {
            'gradient': cond_grad,
            'scale': args.classifier_scale,
            'cls': cls
        }

        # a few lines of code to apply EDS
        if args.use_entropy_scale:
            with th.no_grad():
                # probs = F.softmax(logits, dim=-1)  # (B, C)
                entropy = entropy_logits  # (B,)
                entropy_scale = 1.0 / (entropy / np.log(NUM_CLASSES))  # (B,)
                entropy_scale = entropy_scale.reshape(-1, 1, 1, 1).repeat(1, *cond_grad[0].shape)
                guidance['scale'] = guidance['scale'] * entropy_scale

        return guidance, new_y


    def model_fn(x, t, y=None, cls=None):
        assert y is not None
        return model(x, t, cls if args.class_cond else None)

    logger.log("Loading similarity matrix")
    sim_matrix = np.load("models/IM_ls_w2v.npz")['arr_0']
    sim_matrix = torch.from_numpy(sim_matrix).to(dist_util.dev())

    logger.log("Looking for previous file")
    checkpoint = os.path.join(output_images_folder, "samples_last.npz")
    checkpoint_temp = os.path.join(output_images_folder, "samples_temp.npz")
    vis_images_folder = os.path.join(output_images_folder, "sample_images")
    os.makedirs(vis_images_folder, exist_ok=True)
    final_file = os.path.join(output_images_folder,
                              f"samples_{args.num_samples}x{args.image_size}x{args.image_size}x3.npz")
    if os.path.isfile(final_file):
        dist.barrier()
        logger.log("sampling complete")
        return
    if os.path.isfile(checkpoint):
        npzfile = np.load(checkpoint)
        all_images = list(npzfile['arr_0'])
        all_labels = list(npzfile['arr_1'])
    else:
        all_images = []
        all_labels = []
    logger.log(f"Number of current images: {len(all_images)}")
    logger.log("sampling...")
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.specified_class is not None:
            classes = th.randint(
                low=int(args.specified_class), high=int(args.specified_class) + 1, size=(args.batch_size,),
                device=dist_util.dev()
            )
        else:
            classes = th.randint(
                low=0, high=args.num_classes, size=(args.batch_size,), device=dist_util.dev()
            )
        vector_classes = sim_matrix[classes]
        vector_classes_sum = torch.unsqueeze(torch.sum(vector_classes, dim=-1), dim=-1)
        vector_classes = vector_classes/vector_classes_sum
        # print(torch.sum(classes, dim = -1 ))
        model_kwargs["y"] = vector_classes
        model_kwargs["cls"] = classes

        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        out = sample_fn(
            model_fn,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            device=dist_util.dev(),
        )  # (B, 3, H, W)
        sample = out['sample']

        if args.save_imgs_for_visualization and dist.get_rank() == 0 and (
                len(all_images) // dist.get_world_size()) < 10:
            save_img_dir = vis_images_folder
            utils.save_image(
                sample.clamp(-1, 1),
                os.path.join(save_img_dir, "samples_{}.png".format(len(all_images))),
                nrow=4,
                normalize=True,
                range=(-1, 1),
            )

        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)  # (B, H, W, 3)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        gathered_labels = [th.zeros_like(classes) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_labels, classes)
        all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        if dist.get_rank() == 0:
            if hfai.client.receive_suspend_command():
                print("Receive suspend - good luck next run ^^")
                hfai.client.go_suspend()
            logger.log(f"created {len(all_images) * args.batch_size} samples")
            np.savez(checkpoint_temp, np.stack(all_images), np.stack(all_labels))
            if os.path.isfile(checkpoint):
                os.remove(checkpoint)
            os.rename(checkpoint_temp, checkpoint)

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    label_arr = np.concatenate(all_labels, axis=0)
    label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(output_images_folder, f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr, label_arr)
        os.remove(checkpoint)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=50000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        classifier_path="",
        classifier_scale=1.0,
        use_entropy_scale=False,
        num_classes=1000,
        save_imgs_for_visualization=True,
        fix_seed=False,
        specified_class=None,
        logdir="",
        eps=0.1,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def entropy_calculate(probs):
    log_probs = th.log(probs)
    entropy = (-log_probs * probs).sum(dim=-1)  # (B,)
    return entropy

def label_augment(y, classes, device, delta_epsilon):
    delta_y = y.detach() * delta_epsilon
    indices = torch.arange(0, delta_y.shape[0]).to(device)
    zero_hot = torch.ones_like(delta_y).to(device)
    zero_hot[indices, classes] = 0.0

    delta_y = delta_y * zero_hot

    # sum_delta_y = torch.sum(delta_y * zero_hot, dim=1) #no need for * zero_hot here/ * before already
    sum_delta_y = torch.sum(delta_y, dim=1)
    delta_y[indices, classes] = -sum_delta_y
    new_y = y - delta_y
    return new_y
    pass


if __name__ == "__main__":
    ngpus = th.cuda.device_count()
    hfai.multiprocessing.spawn(main, args=(), nprocs=ngpus, bind_numa=True)
