try:
    from diffusers import StableDiffusionInpaintPipeline
    from transformers import pipeline
except:
    pass
import torch
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import os.path as path
import numpy as np
from tqdm import tqdm
import os

def draw_grid(image):
    plt.imshow(image)
    plt.grid(True)
    plt.savefig("axes.png")

def get_depth(image):
    if path.exists("depth.pt"):
        depth = torch.load("depth.pt")
    else:
        pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
        output = pipe(image)
        depth_img = output["depth"]
        depth = output["predicted_depth"]
        print(depth.min(), depth.max())
        depth = 1 / (0.5 + depth)
        depth_img.save("depth.png")
        torch.save(depth, "depth.pt")
    return np.asarray(depth)

def test_depth(depth, ori):
    ori = ori.astype(depth.dtype)
    for i, dd in tqdm(enumerate(range(0, 40, 1))):
        d = dd / 100
        test = np.clip(ori + (depth > d)[..., None] * 100, a_min=None, a_max=255)
        test_img = Image.fromarray(test.astype(np.uint8))
        test_img.save(f"test/{i} - {d}.png")

def inpaint(image, mask, prompt):
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float16,  # Use float16 for GPU efficiency
    )
    pipe = pipe.to("cuda")  # Use "cpu" if no GPU
    
    negative_prompt = "blurry, deformed, ugly"  # Optional: Exclude unwanted artifacts

    output = pipe(
        prompt=prompt,
        image=image,          # Original living room image
        mask_image=mask,      # White area = where cat appears
        num_inference_steps=100,  # More steps = higher quality (slower)
        guidance_scale=7.5,     # Controls prompt adherence
        negative_prompt=negative_prompt,
    ).images[0].resize(image.size)

    return output

if __name__ == "__main__":
    
    working_dir = path.expanduser(".")
    ori_img = Image.open("original.png").convert("RGB")
    ori = np.asarray(ori_img)
    # Size: 2048 * 1152
    depth = get_depth(ori_img)

    # Where cat should be
    crop_area = [0, 500, 300, 900]
    mask_area = [0, 600, 150, 800]
    for i in range(4):
        mask_area[i] -= crop_area[i % 2]

    cropped_img = ori_img.crop(crop_area)
    cropped_img.save("cropped.png")
    cropped = ori[crop_area[1]: crop_area[3], crop_area[0]: crop_area[2]]
    depth_cropped = depth[crop_area[1]: crop_area[3], crop_area[0]: crop_area[2]]

    # Test    
    # test_depth(depth, ori)
    # raise Exception
    
    threshold = 0.15
    
    mask = Image.new("L", cropped_img.size, 0)  # Start with all-black mask
    draw = ImageDraw.Draw(mask)
    draw.rectangle(mask_area, fill=255)
    mask_ = np.asarray(mask)
    mask_ = (mask_ == 255)
    # mask_ |= (depth_cropped <= threshold)
    mask_ = mask_.astype(np.uint8) * 255
    mask = Image.fromarray(mask_)
    mask.save("mask.png")
    
    comb_mask_ = np.clip(cropped + mask_[..., None] * 100, a_max=255, a_min=None).astype(np.uint8)
    comb_mask = Image.fromarray(comb_mask_)
    comb_mask.save("combined.png")
    
    prompt = "a furry black cat with brown ears, side view"
    output = inpaint(cropped_img, mask, prompt)
    output.save("inpainted_cropped_raw.png")
    
    inpainted = np.asarray(output)
    inpainted = inpainted * (depth_cropped > threshold)[..., None] + cropped * (depth_cropped <= threshold)[..., None]
    inpainted_img = Image.fromarray(inpainted)
    inpainted_img.save("inpainted_cropped.png")

    ori_img.paste(inpainted_img, (crop_area[0], crop_area[1]))
    ori_img.save("inpainted_original.png")