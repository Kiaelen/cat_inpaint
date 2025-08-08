from diffusers import StableDiffusionInpaintPipeline
from transformers import pipeline
import torch
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import os.path as path
import numpy as np

def draw_grid(image):
    plt.imshow(image)
    plt.grid(True)
    plt.savefig("axes.png")

def get_depth(image):
    pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
    depth = pipe(image)["depth"]
    depth.save("depth.png")

if __name__ == "__main__":
    
    working_dir = path.expanduser(".")

    ori_img = Image.open(path.join(working_dir, "original.png")).convert("RGB")
    ori = np.asarray(ori_img)
    # Size: 2048 * 1152
    depth_img = Image.open(path.join(working_dir, "depth.png"))
    depth = np.asarray(depth_img)

    # Where cat should be
    crop_area = np.array([0, 0, 512, 512])
    mask_area = np.array([100, 100, 356, 356])

    for i in range(4):
        mask_area[i] -= crop_area[i % 2]

    cropped_img = ori_img.crop(crop_area)
    cropped_img.save("cropped.png")
    cropped = ori[crop_area[0]: crop_area[2], crop_area[1]: crop_area[3]]
    depth_cropped = depth[crop_area[0]: crop_area[2], crop_area[1]: crop_area[3]]

    # Test
    filtered = cropped * (depth_cropped < 125)[..., None]
    filtered_img = Image.fromarray(filtered)
    filtered_img.save("filtered.png")
    
    # raise Exception


    # mask = Image.new("L", cropped.size, 0)  # Start with all-black mask
    # draw = ImageDraw.Draw(mask)
    # draw.rectangle(mask_area, fill=255)
    # mask.save("mask.png")  # Optional: Save for verification


    # pipe = StableDiffusionInpaintPipeline.from_pretrained(
    #     "stabilityai/stable-diffusion-2-inpainting",
    #     torch_dtype=torch.float16,  # Use float16 for GPU efficiency
    # )
    # pipe = pipe.to("cuda")  # Use "cpu" if no GPU

    # prompt = "an obscure small black cat sitting on the shelf"  # Customize based on location (sofa/table)
    # negative_prompt = "blurry, deformed, ugly"  # Optional: Exclude unwanted artifacts

    # output = pipe(
    #     prompt=prompt,
    #     image=image,          # Original living room image
    #     mask_image=mask,      # White area = where cat appears
    #     num_inference_steps=100,  # More steps = higher quality (slower)
    #     guidance_scale=7.5,     # Controls prompt adherence
    #     negative_prompt=negative_prompt,
    # ).images[0].resize(image.size)

    # output.save("inpainted_cropped.jpg")

    # ori_img.paste(output, (crop_area[0], crop_area[1]))
    # ori_img.save("inpainted_original.png")