from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
from diffusers.utils import load_image, make_image_grid
from PIL import Image
import numpy as np
import torch
import torchvision
from pathlib import Path
from tqdm import tqdm
from quick_calib.camera import IMAGE_H, IMAGE_W

SD_WEIGHTS = "stabilityai/stable-diffusion-xl-base-1.0"
CONTROLNET_WEIGHTS = "diffusers/controlnet-canny-sdxl-1.0"
ROOT_DIR = Path("../data/synth_road")
NEGATIVE_PROMPT = "sketches" # maybe not relevant?
BASE_PROMPT = "dashcam footage of a road"
ROAD_MODIFIERS = ["", "", "", ", snowy road", ", downtown san francisco", ", highway", ", country side", ", dirt road"]
WEATHER_MODIFIERS = ["", "", "", ", clear weather", ", cloudy weather", ", rainy weather", ", stormy weather", ", snowy weather"]


def diffuse(pipe, canny_image, prompt):
    image = pipe(
        prompt,
        negative_prompt=NEGATIVE_PROMPT,
        height = 896,
        width = 1152,
        num_inference_steps = 30,
        image=canny_image,
        controlnet_conditioning_scale=0.5,
    ).images[0]

    return image


def gen(pipe):
    # make output dir
    output_dir = ROOT_DIR / "images"
    output_dir.mkdir(exist_ok=True) # ok to override. TODO add a warning

    files = [file for file in ROOT_DIR.glob("canny/*.png")]

    # iterate over png files in root_dir / canny
    for file in tqdm(files):
        canny_image = Image.open(file)
        canny_image = torchvision.transforms.functional.resize(canny_image, (896, 1152))

        # make prompt
        prompt = BASE_PROMPT + np.random.choice(ROAD_MODIFIERS) + np.random.choice(WEATHER_MODIFIERS)
        print(prompt)

        # diffuse, resize and save
        image = diffuse(pipe, canny_image, prompt)

        image = torchvision.transforms.functional.resize(image, (IMAGE_H, IMAGE_W))
        image.save(output_dir / file.name)


if __name__ == "__main__":
    controlnet = ControlNetModel.from_pretrained(
        CONTROLNET_WEIGHTS,
        torch_dtype=torch.float16,
        use_safetensors=True
    )

    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16, use_safetensors=True)

    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        SD_WEIGHTS,
        controlnet=controlnet,
        vae=vae,
        torch_dtype=torch.float16,
        use_safetensors=True
    )
    pipe.enable_model_cpu_offload()

    gen(pipe)
