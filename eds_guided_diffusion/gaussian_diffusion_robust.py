"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""
import copy

# import torch

from eds_guided_diffusion.gaussian_diffusion import *
from eds_guided_diffusion.respace import *
import random
import torch
import torch.nn as nn
import torch.nn.functional


class GaussianDiffusionLSimSchedule(SpacedDiffusion):
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
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
    ):
        super(GaussianDiffusionLSimSchedule, self).__init__(use_timesteps=use_timesteps, betas=betas,
                                                   model_mean_type=model_mean_type,
                                                   model_var_type=model_var_type,
                                                   loss_type=loss_type,
                                                   rescale_timesteps=rescale_timesteps)
        self.epsilon = 0.1

    def p_sample(
            self,
            model,
            x,
            t,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None
    ):
        """
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
            out = self.condition_mean(
                cond_fn, out, x, t, model_kwargs=model_kwargs
            )

        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop_progressive(
            self,
            model,
            shape,
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            device=None,
            progress=False
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():

                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                )

                y = self.label_augment(model_kwargs["y"], model_kwargs["cls"], device)
                model_kwargs["y"] = y
                yield out
                img = out["sample"]

    def label_augment(self, y, classes, device):
        new_y = y.detach() * self.epsilon
        indices = torch.arange(0, new_y.shape[0]).to(device)
        new_y[indices, classes] = 1.0
        sum_new_y = torch.unsqueeze(torch.sum(new_y, dim=-1), dim=-1)
        return new_y/sum_new_y
        pass


class GaussianDiffusionLSimScheduleWconf(GaussianDiffusionLSimSchedule):
    def __init__(self,
        use_timesteps,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
    ):
        super(GaussianDiffusionLSimScheduleWconf, self).__init__(use_timesteps=use_timesteps, betas=betas,
                                                   model_mean_type=model_mean_type,
                                                   model_var_type=model_var_type,
                                                   loss_type=loss_type,
                                                   rescale_timesteps=rescale_timesteps)
        self.epsilon = 0.1

    def label_augment(self, y, classes, device):
        delta_epsilon = 1 - self.epsilon
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

class GaussianDiffusionLSimScheduleAdaptWconf(SpacedDiffusion):
    def __init__(self,
        use_timesteps,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
    ):
        super(GaussianDiffusionLSimScheduleAdaptWconf, self).__init__(use_timesteps=use_timesteps, betas=betas,
                                                   model_mean_type=model_mean_type,
                                                   model_var_type=model_var_type,
                                                   loss_type=loss_type,
                                                   rescale_timesteps=rescale_timesteps)
        self.epsilon = 0.1

    # def label_augment(self, y, classes, device):
    #     delta_epsilon = 1 - self.epsilon
    #     delta_y = y.detach() * delta_epsilon
    #     indices = torch.arange(0, delta_y.shape[0]).to(device)
    #     zero_hot = torch.ones_like(delta_y).to(device)
    #     zero_hot[indices, classes] = 0.0
    #
    #     delta_y = delta_y * zero_hot
    #
    #     # sum_delta_y = torch.sum(delta_y * zero_hot, dim=1) #no need for * zero_hot here/ * before already
    #     sum_delta_y = torch.sum(delta_y, dim=1)
    #     delta_y[indices, classes] = -sum_delta_y
    #     new_y = y - delta_y
    #     return new_y
    #     pass

    def condition_mean_mod_y(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
                Compute the mean for the previous step, given a function cond_fn that
                computes the gradient of a conditional log probability with respect to
                x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
                condition on y.

                This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
                """
        guidance, new_y = cond_fn(x, self._scale_timesteps(t), **model_kwargs)

        model_kwargs['y'] = new_y # need to check if model_kwargs['y'] change actually

        p_mean_var['mean'] = (
                p_mean_var["mean"].float() + p_mean_var["variance"] * guidance['scale'] * guidance['gradient'].float()
        )
        return p_mean_var

    def condition_mean(self, cond_fn, *args, **kwargs):
        return self.condition_mean_mod_y(self._wrap_model(cond_fn), *args, **kwargs)

class GaussianDiffusionOFS(SpacedDiffusion):
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
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
    ):
        super(GaussianDiffusionOFS, self).__init__(use_timesteps=use_timesteps, betas=betas,
                                                   model_mean_type=model_mean_type,
                                                   model_var_type=model_var_type,
                                                   loss_type=loss_type,
                                                   rescale_timesteps=rescale_timesteps)
        self.epsilon = 0.1

    def p_sample(
            self,
            model,
            x,
            t,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None
    ):
        """
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
            out = self.condition_mean(
                cond_fn, out, x, t, model_kwargs=model_kwargs
            )

        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop_progressive(
            self,
            model,
            shape,
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            device=None,
            progress=False
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():

                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                )

                yield out
                img = out["sample"]

    def condition_mean_ofs(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        guidance = cond_fn(x, self._scale_timesteps(t), **model_kwargs)

        p_mean_var['mean'] = (
                    p_mean_var["mean"].float() + guidance['scale'] * guidance['gradient'].float()
            )
        return p_mean_var

    def condition_mean(self, cond_fn, *args, **kwargs):
        return self.condition_mean_ofs(self._wrap_model(cond_fn), *args, **kwargs)


class GaussianDiffusionOFS_X0Pred(SpacedDiffusion):
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
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
    ):
        super(GaussianDiffusionOFS_X0Pred, self).__init__(use_timesteps=use_timesteps, betas=betas,
                                                   model_mean_type=model_mean_type,
                                                   model_var_type=model_var_type,
                                                   loss_type=loss_type,
                                                   rescale_timesteps=rescale_timesteps)
        self.epsilon = 0.1

    def p_sample(
            self,
            model,
            x,
            t,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None
    ):
        """
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
            out = self.condition_mean(
                cond_fn, out, x, t, model_kwargs=model_kwargs
            )

        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop_progressive(
            self,
            model,
            shape,
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            device=None,
            progress=False
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():

                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                )

                yield out
                img = out["sample"]

    def condition_mean_ofs(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        t_0 = torch.zeros_like(t, device=t.get_device())
        guidance = cond_fn(p_mean_var['pred_xstart'], self._scale_timesteps(t_0), **model_kwargs)

        p_mean_var['mean'] = (
                    p_mean_var["mean"].float() + p_mean_var["variance"] * guidance['scale'] * guidance['gradient'].float()
            )
        return p_mean_var

    def condition_mean(self, cond_fn, *args, **kwargs):
        return self.condition_mean_ofs(self._wrap_model(cond_fn), *args, **kwargs)

