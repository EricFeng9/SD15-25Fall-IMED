#!/usr/bin/env python
# -*- coding: utf-8 -*-
print("脚本开始执行...")
import sys
print(f"Python 版本: {sys.version}")

import os
print("设置环境变量...")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["DISABLE_TELEMETRY"] = "1"

print("导入基础模块...")
import csv, math, itertools
import torch, numpy as np
import cv2
from PIL import Image
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

print("导入 diffusers...")
from diffusers import (DDPMScheduler, StableDiffusionControlNetPipeline,
                       ControlNetModel, 
                       AutoencoderKL, UNet2DConditionModel)

print("导入 transformers...")
from transformers import CLIPTextModel, CLIPTokenizer

print("导入 pytorch_msssim...")
from pytorch_msssim import MS_SSIM

print("导入 time, argparse...")
import time
import argparse

print("导入 data_loader_all...")
from data_loader_all import (
    UnifiedDataset, SIZE, preprocess_for_vessel_extraction,
    GAMMA_CFFA, GAMMA_CFOCTA_CF, GAMMA_CFOCTA_OCTA,
    GAMMA_CFOCT_CF, GAMMA_CFOCT_OCT, FRANGI_SIGMAS, FRANGI_ALPHA, FRANGI_BETA,
    create_eroded_mask,
    get_image_params
)

print("导入 registration_cf_octa...")
from registration_cf_octa import load_affine_matrix, apply_affine_registration

print("所有导入完成！")
