import os
from PIL import Image
import cv2
import numpy as np

def preprocess_dataset(input_dir, output_dir, size=512):
    os.makedirs(output_dir, exist_ok=True)
    for img_name in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        img = img.resize((size, size), Image.LANCZOS)
        img.save(os.path.join(output_dir, img_name))
