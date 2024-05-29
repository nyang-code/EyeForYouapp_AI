import torch
from transformers import pipeline
from PIL import Image
import numpy as np
import os
from tqdm import tqdm

def remove_background(images, device, pipe):
    results = []
    for image in images:
        pillow_mask = pipe(image, return_mask=True)
        pillow_image = pipe(image)
        
        np_image = np.array(pillow_image)
        np_image[np_image == 0] = 255  # 배경을 하얀색으로 설정
        
        modified_image = Image.fromarray(np_image)
        if modified_image.mode == 'RGBA':
            modified_image = modified_image.convert('RGB')
        
        results.append(modified_image)
    return results

def process_images_in_batches(input_folder, output_folder, batch_size=10):
    device = torch.device("cuda:3") if torch.cuda.is_available() else torch.device("cpu")
    pipe = pipeline("image-segmentation", model="briaai/RMBG-1.4", device=device, trust_remote_code=True)
    
    files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]
    batches = [files[i:i + batch_size] for i in range(0, len(files), batch_size)]
    
    for batch in tqdm(batches):
        image_paths = [os.path.join(input_folder, filename) for filename in batch]
        images = [Image.open(path) for path in image_paths]
        result_images = remove_background(images, device, pipe)
        
        for result_image, filename in zip(result_images, batch):
            output_path = os.path.join(output_folder, filename)  # 결과 파일 경로 생성
            result_image.save(output_path)  # 이미지 저장

# 경로 설정
input_folder = '/home/mmc/disk2/duck/cap/data/snack/train/images'
output_folder = '/home/mmc/disk2/duck/cap/data/background_snack_mix/'

# 배치 단위 이미지 처리 실행
process_images_in_batches(input_folder, output_folder)
