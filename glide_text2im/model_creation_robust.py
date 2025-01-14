from glide_text2im.model_creation import *
from glide_text2im.gaussian_diffusion_robust import *


def create_model_and_diffusion_infodeg(
    image_size,
    num_channels,
    num_res_blocks,
    channel_mult,
    num_heads,
    num_head_channels,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    text_ctx,
    xf_width,
    xf_layers,
    xf_heads,
    xf_final_ln,
    xf_padding,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_scale_shift_norm,
    resblock_updown,
    use_fp16,
    cache_text_emb,
    inpaint,
    super_res,
):
    model = create_model(
        image_size,
        num_channels,
        num_res_blocks,
        channel_mult=channel_mult,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        text_ctx=text_ctx,
        xf_width=xf_width,
        xf_layers=xf_layers,
        xf_heads=xf_heads,
        xf_final_ln=xf_final_ln,
        xf_padding=xf_padding,
        resblock_updown=resblock_updown,
        use_fp16=use_fp16,
        cache_text_emb=cache_text_emb,
        inpaint=inpaint,
        super_res=super_res,
    )
    diffusion = create_gaussian_diffusion_infodeg(
        steps=diffusion_steps,
        noise_schedule=noise_schedule,
        timestep_respacing=timestep_respacing,
    )
    return model, diffusion


def create_gaussian_diffusion_infodeg(
    steps,
    noise_schedule,
    timestep_respacing,
):
    betas = get_named_beta_schedule(noise_schedule, steps)
    if not timestep_respacing:
        timestep_respacing = [steps]
    return GaussianDiffusionRobust(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
    )

def create_model_and_diffusion_prog(
    image_size,
    num_channels,
    num_res_blocks,
    channel_mult,
    num_heads,
    num_head_channels,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    text_ctx,
    xf_width,
    xf_layers,
    xf_heads,
    xf_final_ln,
    xf_padding,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_scale_shift_norm,
    resblock_updown,
    use_fp16,
    cache_text_emb,
    inpaint,
    super_res,
):
    model = create_model(
        image_size,
        num_channels,
        num_res_blocks,
        channel_mult=channel_mult,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        text_ctx=text_ctx,
        xf_width=xf_width,
        xf_layers=xf_layers,
        xf_heads=xf_heads,
        xf_final_ln=xf_final_ln,
        xf_padding=xf_padding,
        resblock_updown=resblock_updown,
        use_fp16=use_fp16,
        cache_text_emb=cache_text_emb,
        inpaint=inpaint,
        super_res=super_res,
    )
    diffusion = create_gaussian_diffusion_prog(
        steps=diffusion_steps,
        noise_schedule=noise_schedule,
        timestep_respacing=timestep_respacing,
    )
    return model, diffusion


def create_gaussian_diffusion_prog(
    steps,
    noise_schedule,
    timestep_respacing,
):
    betas = get_named_beta_schedule(noise_schedule, steps)
    if not timestep_respacing:
        timestep_respacing = [steps]
    return GaussianDiffusionProG(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
    )

