from libraries import *
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from torchvision.utils import draw_bounding_boxes
import numpy as np
from PIL import ImageGrab
import cv2
import time
import pyautogui
from numpy import ones,vstack
from numpy.linalg import lstsq
from statistics import mean



class YOLO:
    def __init__(self, device, conf = 0.5, min_distance = (400, 200), brake_intensity = 0.6):
        self.device = device
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.model.to(device)
        self.model.conf = conf
        self.model.classes = [0, 1, 2, 3, 5, 7]
        
        self.classes = {2 : "car", 0 : "person", 7 : "truck", 1 : "bicycle", 3 : "motorcycle", 5 : "bus"}
        
        self.min_distance = min_distance
        self.brake_intensity = brake_intensity
    
    def _euclideanDistance(self, coordinates):
        return pow(pow(coordinates[0] - coordinates[2], 2) + pow(coordinates[1] - coordinates[3], 2), .5)
    
    
    
    def _brake(self, boxes, labels):
        
        brake = 0.0
        
        try:
            d = np.apply_along_axis(self._euclideanDistance, 1, boxes)
            d = np.expand_dims(d, 1)
            labels = np.expand_dims(labels, 1)
            
            p = np.concatenate([d, labels], axis =1)
            
            if(((p[:, 0] > self.min_distance[0]) & ((p[:, 1] == 2) | (p[:, 1] == 7) | (p[:, 1] == 5))).any()):
                brake = self.brake_intensity
            elif(((p[:, 0] > self.min_distance[1]) & ((p[:, 1] == 0) | (p[:, 1] == 1) | (p[:, 1] == 3))).any()):
                brake = self.brake_intensity
        
        except ValueError as e:
            pass
        
                    
                
        return brake
    
    def forward(self, image):
        
        results = self.model(image)
        boxes = results.xyxy[0][:, :4].cpu().numpy()
        labels = results.xyxy[0][:, 5].cpu().to(dtype = torch.int32).numpy()
        
        
        
        brake = self._brake(boxes, labels)
        
        results.render()
        
        return results.ims[0], brake
        

