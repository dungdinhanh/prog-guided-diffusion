import os
import tempfile

import numpy as np
import torchvision
from tqdm.auto import tqdm
from evaluations.evaluator_tolog import *
import tensorflow.compat.v1 as tf

CLASSES = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def main():
    for split in ["train", "test"]:
        out_dir = f"reference"

        print("downloading...")

        dataset = torchvision.datasets.CIFAR10(
            root="data/CIFAR", train=(split == "train"), download=True
        )

        print("dumping images...")
        os.makedirs(out_dir, exist_ok=True)
        list_images = []
        list_targets = []
        for i in tqdm(range(len(dataset))):
            image, label = dataset[i]
            arr_image = np.asarray(image).astype(np.uint8)
            arr_label = np.asarray(label)
            list_images.append(arr_image)
            list_targets.append(arr_label)
        list_images = np.stack(list_images)
        list_targets = np.stack(list_targets)
        save_file = os.path.join(out_dir, f"VIRTUAL_cifar10_labeled_{split}.npz")
        np.savez(save_file, list_images, list_targets)

        # config = tf.ConfigProto(
        #     allow_soft_placement=True  # allows DecodeJpeg to run on CPU in Inception graph
        # )
        # config.gpu_options.allow_growth = True
        # evaluator = Evaluator(tf.Session(config=config))
        #
        # print("computing reference batch activations...")
        # ref_acts = evaluator.read_activations(save_file)
        # print("computing/reading reference batch statistics...")
        # ref_stats, ref_stats_spatial = evaluator.read_statistics(save_file, ref_acts)
        # np.savez(save_file, list_images, mu=ref_stats.mu, sigma=ref_stats,
        #          mu_s=ref_stats_spatial.mu, sigma_s=ref_stats.sigma, allow_pickle=True)





        # reference / VIRTUAL_cifar10_labeled.npz
    # test = np.load("reference/VIRTUAL_imagenet64_labeled.npz")
    # # print(list(test.keys()))
    # arr1 = test['arr_0']
    # print(arr1.shape)
    # # print(arr2.shape)

if __name__ == "__main__":
    main()
