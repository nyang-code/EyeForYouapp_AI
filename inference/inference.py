import torch
from transformers import pipeline
from PIL import Image
import numpy as np
from ultralytics import YOLO


def remove_background(image_path):
    device = torch.device("cuda:2") if torch.cuda.is_available() else torch.device("cpu")
    pipe = pipeline("image-segmentation", model="briaai/RMBG-1.4", device=device, trust_remote_code=True)

    pillow_mask = pipe(image_path, return_mask=True)
    pillow_image = pipe(image_path)

    np_image = np.array(pillow_image)

    np_image[np_image == 0] = 255

    modified_image = Image.fromarray(np_image)

    if modified_image.mode == 'RGBA':
        modified_image = modified_image.convert('RGB')

    return modified_image


def predict_image(pil_image, yolo_model_path, device=2):
    model = YOLO(yolo_model_path)
    results = model.predict(pil_image, save=False, device=device)
    return results


def process_image(image_path, yolo_model_path):
    modified_image = remove_background(image_path)
    results = predict_image(modified_image, yolo_model_path)
    return results


image_path = "/home/mmc/disk2/duck/cap/ultralytics/coca_.jpg"
yolo_model_path = '/home/mmc/disk2/duck/cap/pt/drink/weights/kang_weight/best.pt'

result_image = process_image(image_path, yolo_model_path)
