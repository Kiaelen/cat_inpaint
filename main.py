from diffusers import StableDiffusionInpaintPipeline
import torch
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt

def draw_grid(image):
    plt.imshow(image)
    plt.grid(True)
    plt.savefig("one-cat_axes")

image = Image.open("/home/yz3397/cat_inpaint/one-cat.png").convert("RGB")
# Size: 2048 * 1152

crop_area = (0, 0, 512, 512)
image = image.crop(crop_area)

print(image.size)
image.save("crop_img.png")

mask = Image.new("L", image.size, 0)  # Start with all-black mask
draw = ImageDraw.Draw(mask)

# Example: Draw a white rectangle where the cat should be (customize coordinates)
sofa_area = (0, 0, 256, 256)  # Replace with your sofa/table coordinates
draw.rectangle(sofa_area, fill=255)

mask.save("mask.png")  # Optional: Save for verification

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float16,  # Use float16 for GPU efficiency
)
pipe = pipe.to("cuda")  # Use "cpu" if no GPU

prompt = "a black cat"  # Customize based on location (sofa/table)
negative_prompt = "blurry, deformed, ugly"  # Optional: Exclude unwanted artifacts

output = pipe(
    prompt=prompt,
    image=image,          # Original living room image
    mask_image=mask,      # White area = where cat appears
    num_inference_steps=50,  # More steps = higher quality (slower)
    guidance_scale=7.5,     # Controls prompt adherence
    negative_prompt=negative_prompt,
).images[0].resize(image.size)

output.save("living_room_with_cat.jpg")