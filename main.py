from diffusers import StableDiffusionInpaintPipeline
import torch
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import os.path as path

def draw_grid(image):
    plt.imshow(image)
    plt.grid(True)
    plt.savefig("one-cat_axes")

ori_image = Image.open(path.join(path.expanduser("."), "one-cat.png")).convert("RGB")
# Size: 2048 * 1152

# Where cat should be
crop_area = (0, 0, 512, 512)
image = ori_image.crop(crop_area)

print(image.size)
image.save("crop_img.png")

mask = Image.new("L", image.size, 0)  # Start with all-black mask
draw = ImageDraw.Draw(mask)

# Mask
sofa_area = (0, 0, 256, 256) 
draw.rectangle(sofa_area, fill=255)

mask.save("mask.png")  # Optional: Save for verification

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float16,  # Use float16 for GPU efficiency
)
pipe = pipe.to("cuda")  # Use "cpu" if no GPU

prompt = "an small black cat sitting on the shelf"  # Customize based on location (sofa/table)
negative_prompt = "blurry, deformed, ugly"  # Optional: Exclude unwanted artifacts

output = pipe(
    prompt=prompt,
    image=image,          # Original living room image
    mask_image=mask,      # White area = where cat appears
    num_inference_steps=100,  # More steps = higher quality (slower)
    guidance_scale=7.5,     # Controls prompt adherence
    negative_prompt=negative_prompt,
).images[0].resize(image.size)

output.save("living_room_with_cat.jpg")

ori_image.paste(output, crop_area[0], crop_area[2])
ori_image.save("altered.png")