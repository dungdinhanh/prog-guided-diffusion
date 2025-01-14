from .model_creation import *


@lru_cache()
def default_config_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")


@attr.s
class CLIPRobustModel(CLIPModel):
    def __init__(self):
        self.lvae = None

    def cond_fn(self, prompts: List[str], grad_scale: float) -> Callable[..., torch.Tensor]:
        with torch.no_grad():
            z_t = self.text_embeddings(prompts)
        assert self.lvae is not None, print("label vae can not be None")

        def cond_fn(x, t, grad_scale=grad_scale, **kwargs):
            with torch.enable_grad():
                print("going here")
                x_var = x.detach().requires_grad_(True)
                z_i = self.image_embeddings(x_var, t)
                t_z_t = self.lvae.forward_encoder(x_var, timesteps=t, labels=z_t)
                loss = torch.exp(self.logit_scale) * (t_z_t * z_i).sum()
                grad = torch.autograd.grad(loss, x_var)[0].detach()
            return grad * grad_scale

        return cond_fn


def create_clip_robust_model(
    config_path: Optional[str] = None,
    device: Optional[torch.device] = None,
    tokenizer: Optional[SimpleTokenizer] = None,
) -> CLIPRobustModel:
    if config_path is None:
        config_path = default_config_path()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if tokenizer is None:
        tokenizer = SimpleTokenizer()

    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    text_encoder = TextEncoder(
        n_bpe_vocab=config["n_vocab"],
        max_text_len=config["max_text_len"],
        n_embd=config["n_embd"],
        n_head=config["n_head_text"],
        n_xf_blocks=config["n_xf_blocks_text"],
        n_head_state=config["n_head_state_text"],
        device=device,
    )

    image_encoder = ImageEncoder(
        image_size=config["image_size"],
        patch_size=config["patch_size"],
        n_embd=config["n_embd"],
        n_head=config["n_head_image"],
        n_xf_blocks=config["n_xf_blocks_image"],
        n_head_state=config["n_head_state_image"],
        n_timestep=config["n_timesteps"],
        device=device,
    )

    logit_scale = torch.tensor(
        np.log(config["logit_scale"]),
        dtype=torch.float32,
        device=device,
        requires_grad=False,
    )

    return CLIPRobustModel(
        config=config,
        text_encoder=text_encoder,
        image_encoder=image_encoder,
        logit_scale=logit_scale,
        device=device,
        tokenizer=tokenizer,
    )
