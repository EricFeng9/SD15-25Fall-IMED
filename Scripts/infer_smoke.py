# /data/student/Fengjunming/SDXL_ControlNet/infer_smoke.py
import os, csv, torch
from PIL import Image
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel

os.environ["HF_HUB_OFFLINE"]="1"
base_dir = "/data/student/Fengjunming/SDXL_ControlNet/models/sdxl-base"
ctrl_dir = "/data/student/Fengjunming/SDXL_ControlNet/models/controlnet-canny-sdxl"
csv_path = "/data/student/Fengjunming/SDXL_ControlNet/Scripts/test_pairs.csv"

with open(csv_path) as f:
    row = next(csv.DictReader(f))
cond = Image.open(row["cond_path"]).convert("RGB").resize((768,768))

controlnet = ControlNetModel.from_pretrained(ctrl_dir, torch_dtype=torch.float16, local_files_only=True)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    base_dir, controlnet=controlnet, torch_dtype=torch.float16, local_files_only=True
).to("cuda")

img = pipe(prompt="", image=cond, num_inference_steps=30, guidance_scale=5.0).images[0]
out = "/data/student/Fengjunming/SDXL_ControlNet/pred_smoke.png"
img.save(out); print("saved:", out)