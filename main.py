from tracemalloc import start
import uvicorn
import cv2
import time
import config
import numpy as np
from model import models,detect
from model.Utils import read_image
from fastapi import FastAPI, File, UploadFile, HTTPException,status

secTech = FastAPI()

model = models.load_model(
    config.YOLOV3_CFG_PATH, 
    config.YOLOV3_WEIGHTS_PATH)
modelTiny = models.load_model(
    config.YOLOV3_TINY_CFG_PATH, 
    config.YOLOV3_TINY_WEIGHTS_PATH)
classes = []
with open(config.LABELS_PATH,'r') as f:
    classes = [x[:len(x)-1] for x in f.readlines()]
@secTech.get('/')
def index():
    return {'message' : 'Welcome To SecTech'}

@secTech.post('/image')
async def imageObjectDetection(frame: str,fast: bool,file: bytes = File(...)):
    startTime = time.time()
    nparr = np.frombuffer(file, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    boxes = detect.detect_image(modelTiny if fast else model, image).tolist()
    response = {'frame' : frame, 'numBoxes' : len(boxes)}
    for i in range(len(boxes)):
        boxes[i] = [int(x) for x in boxes[i]]
        boxes[i][-1] = classes[boxes[i][-1]]
        response['box'+ str(i)] = str(boxes[i])
    response['processTime'] = str(int(1000*(time.time() - startTime))) + " ms"
    return response

if __name__ == '__main__':
    uvicorn.run(secTech, host='127.0.0.1', port = 8000)