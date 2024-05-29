import torch
from transformers import pipeline
from PIL import Image
import numpy as np
import os
from tqdm import tqdm

def remove_background(image_path, device, pipe):
    pillow_mask = pipe(image_path, return_mask=True)
    pillow_image = pipe(image_path)

    np_image = np.array(pillow_image)
    np_image[np_image == 0] = 255

    modified_image = Image.fromarray(np_image)
    if modified_image.mode == 'RGBA':
        modified_image = modified_image.convert('RGB')

    return modified_image

def process_images(input_folder, output_folder):
    device = torch.device("cuda:3") if torch.cuda.is_available() else torch.device("cpu")
    pipe = pipeline("image-segmentation", model="briaai/RMBG-1.4", device=device, trust_remote_code=True)

    # 입력 폴더에서 모든 파일 목록을 가져옴
    for filename in tqdm(os.listdir(input_folder)):
        if filename.endswith(('.jpg', '.png', '.jpeg')):  # 이미지 파일 형식 필터링
            image_path = os.path.join(input_folder, filename)
            result_image = remove_background(image_path, device, pipe)
            output_path = os.path.join(output_folder, filename)  # 결과 파일 경로 생성
            result_image.save(output_path)  # 이미지 저장

# 경로 설정
input_folder = '/home/mmc/disk2/duck/cap/data/noodle/train/images'
output_folder = '/home/mmc/disk2/duck/cap/data/background_noodle_mix/'

# 이미지 처리 실행
process_images(input_folder, output_folder)
