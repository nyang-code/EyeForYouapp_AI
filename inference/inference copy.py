from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from transformers import pipeline
import torch
from PIL import Image
import numpy as np
from ultralytics import YOLO

app = FastAPI()

def remove_background(image):
    device = torch.device("cpu")
    pipe = pipeline("image-segmentation", model="briaai/RMBG-1.4", device=device, trust_remote_code=True)
    pillow_mask = pipe(image, return_mask=True)
    pillow_image = pipe(image)
    
    np_image = np.array(pillow_image)
    np_image[np_image == 0] = 255
    
    modified_image = Image.fromarray(np_image)
    if modified_image.mode == 'RGBA':
        modified_image = modified_image.convert('RGB')
    
    return modified_image

def predict_image(pil_image, yolo_model_path):
    model = YOLO(yolo_model_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    results = model(pil_image)
    results_object = results[0] 

    # 감지된 객체가 있는지 확인
    if not results_object.boxes or len(results_object.boxes.data) == 0:
        object_names = ["감지된 객체 없음"]
    else:
        object_names = [results_object.names[int(cls)] for cls in results_object.boxes.cls]

    return object_names




@app.post("/upload/")
async def process_image(file: UploadFile = File(...)):
    try:
        yolo_model_path = '/home/youjin/다운로드/best.pt'
        image = Image.open(file.file)
        modified_image = remove_background(image),
        object_names = predict_image(modified_image, yolo_model_path)

        # Format the response in a user-friendly way
        response_data = object_names
        return JSONResponse(content=response_data, headers={"Content-Type" : "application/json; charset=utf-8"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"An error occurred: {str(e)}"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)