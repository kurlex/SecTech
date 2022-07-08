import uvicorn
from fastapi import FastAPI

secTech = FastAPI()

@secTech.get('/')
def index():
    return {'message' : 'Welcome To SecTech'}

@secTech.get('/image')
def imageObjectDetection():
    return {'message' : 'upload your image'}

@secTech.get('/video')
def videoObjectDetection():
    return {'message' : 'upload your video'}

@secTech.get('/live')
def liveObjectDetection():
    return {'message' : 'Welcome To live object detection'}


if __name__ == '__main__':
    uvicorn.run(secTech, host='127.0.0.1', port = 8000)