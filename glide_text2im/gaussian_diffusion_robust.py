from glide_text2im.respace import *
import random
import torch
import torch.nn as nn
import torch.nn.functional


class GaussianDiffusionRobust(SpacedDiffusion):
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(self,
        use_timesteps,
        *,
        betas
    ):
        super(GaussianDiffusionRobust, self).__init__(use_timesteps=use_timesteps, betas=betas,)
        self.osc = 0.2

    def p_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
    ):
        """
        Batch consider

        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = th.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        if cond_fn is not None:
            out["mean"] = self.condition_mean(cond_fn, out, x, t ,model_kwargs=model_kwargs)
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def condition_mean(self, cond_fn, *args, **kwargs):
        return self.condition_mean_robust(self._wrap_model(cond_fn), *args, **kwargs)

    def condition_mean_robust(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
                Compute the mean for the previous step, given a function cond_fn that
                computes the gradient of a conditional log probability with respect to
                x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
                condition on y.

                This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
                """
        gradient = cond_fn(x, t, osc=self.osc ,**model_kwargs)
        new_mean = p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
        return new_mean

class GaussianDiffusionProG(GaussianDiffusionRobust):
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(self,
        use_timesteps,
        *,
        betas
    ):
        super(GaussianDiffusionProG, self).__init__(use_timesteps=use_timesteps, betas=betas,)
        self.osc = 0.2
        self.eps = 0.9

    def condition_mean_robust(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
                Compute the mean for the previous step, given a function cond_fn that
                computes the gradient of a conditional log probability with respect to
                x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
                condition on y.

                This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
                """
        gradient = cond_fn(x, t, osc=self.osc,**model_kwargs)
        new_mean = p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
        self.augment_osc()
        return new_mean

    def augment_osc(self):
        all_others = 1 - self.osc
        delta_y = all_others * 0.1
        self.osc += delta_y

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res + th.zeros(broadcast_shape, device=timesteps.device)