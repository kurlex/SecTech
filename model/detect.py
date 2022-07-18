import numpy as np
import torch
import torchvision.transforms as transforms
from model.Utils import rescale_boxes,non_max_suppression
from model.Transforms import Resize, DEFAULT_TRANSFORMS

def detect_image(model, image, img_size=416, conf_thres=0.5, nms_thres=0.5):
    model.eval()
    input_img = transforms.Compose([
        DEFAULT_TRANSFORMS,
        Resize(img_size)])(
            (image, np.zeros((1, 5))))[0].unsqueeze(0)
    if torch.cuda.is_available():
        input_img = input_img.to("cuda")
    with torch.no_grad():
        detections = model(input_img)
        detections = non_max_suppression(detections, conf_thres, nms_thres)
        detections = rescale_boxes(detections[0], img_size, image.shape[:2])
    return detections.numpy()
