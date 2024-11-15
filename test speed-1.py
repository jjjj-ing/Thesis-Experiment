import torch
from diffusers import Transformer2DModel, PixArtSigmaPipeline

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
weight_dtype = torch.float16

pipe = PixArtSigmaPipeline.from_pretrained(
    "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
    torch_dtype=weight_dtype,
    use_safetensors=True,
)
pipe.to(device)

# Enable memory optimizations.
# pipe.enable_model_cpu_offload()

prompt = "A small cactus with a happy face in the Sahara desert."
image = pipe(prompt).images[0]
image.save("./catcus.png")