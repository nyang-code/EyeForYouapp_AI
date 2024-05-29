import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from data_loader import SalObjDataset, RescaleT, ToTensorLab
from model import U2NET
from ultralytics import YOLO
import os
from PIL import Image

def unetr_yolo8(input_image):
    # 모델 로드
    model_dir = '/path/to/u2net.pth'
    net = U2NET(3, 1)
    net.load_state_dict(torch.load(model_dir, map_location=torch.device('cpu')))
    net.eval()

    # 이미지를 텐서로 변환
    transform = transforms.Compose([RescaleT(320), ToTensorLab(flag=0)])
    image_tensor = transform({'image': input_image, 'label': None})['image'].unsqueeze(0)

    # 추론
    d1, _, _, _, _, _, _ = net(image_tensor)
    pred = (d1[:, 0, :, :] - d1.min()) / (d1.max() - d1.min())  # 정규화

    # 출력 처리
    mix_image = (pred.squeeze().cpu().data.numpy() * 255).astype(np.uint8)
    background = cv2.bitwise_not(mix_image)

    white_mask = mix_image >= 200
    white_mask_3d = np.stack([white_mask]*3, axis=-1)

    dir_array = np.array(input_image)
    result_array = np.where(white_mask_3d, dir_array, background)
    # unetr로 배경은 흰색 물체만 보이는 상태
    result_image = Image.fromarray(result_array.astype(np.uint8))

    # YOLO 모델 로드
    yolo_model = YOLO('/home/mmc/disk2/duck/cap/pt/drink/weights/kang_weight/best.pt')

    # 신뢰도 임계값 설정
    conf_threshold = 0.1  # 10% 이상의 신뢰도를 가진 결과만 포함

    # 이미지에 대한 추론 실행
    results = yolo_model(result_image, conf_thres=conf_threshold)
    # 결과 처리 및 출력
    class_ids = []  # 클래스 ID를 저장할 리스트
    for result in results:
        boxes = result.boxes  # 각 바운딩박스의 좌표
        probs = result.probs  # 각 바운딩박스에 대한 클래스 확률
        for prob in probs:
            class_id = prob.argmax()  # 가장 높은 확률을 가진 클래스의 인덱스
            class_ids.append(class_id)  # 클래스 ID를 리스트에 추가

    # 모든 클래스 ID 출력
    for index, class_id in enumerate(class_ids):
        print(f"Class ID {index}: {class_id}")

    return class_ids


