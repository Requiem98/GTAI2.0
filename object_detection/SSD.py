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


class SSD:
    def __init__(self, min_distance = (400, 200), brake_intensity = 0.6):
        self.model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)
        self.model.to("cuda")
        self.model.eval()
        self.preprocess = SSD300_VGG16_Weights.DEFAULT.transforms()
        self.classes = {3 : "car", 1 : "person", 8 : "truck", 4 : "cycle"}
        self.min_distance = min_distance
        self.brake_intensity = brake_intensity
    
    def _euclideanDistance(self, coordinates):
        return pow(pow(coordinates[0] - coordinates[2], 2) + pow(coordinates[1] - coordinates[3], 2), .5)
    
    
    def _remove_boxes(self, prediction, threshold = 0.7):
        
        mask = (prediction['scores'] > threshold) & (
            (prediction['labels'] == 3) | (prediction['labels'] == 1) | (prediction['labels'] == 8) | (prediction['labels'] == 4))
    
        for key,val in prediction.items():
            prediction[key] = val[mask]
            
        return prediction
    
    def _create_labels_and_brake(self, prediction):
        labels = list()
        brake = 0.0
        
        for l, s, b in zip(prediction["labels"].cpu().numpy(), prediction["scores"].cpu().numpy(), prediction["boxes"].cpu().numpy()):
            d = self._euclideanDistance(b)
            try:
                labels.append(f"{self.classes[l]} :: {round(s, 2)} :: {d}")
                
                if((d > self.min_distance[0]) and (l == 3 or l == 8)):
                    brake = self.brake_intensity
                elif((d > self.min_distance[1]) and (l != 3 and l != 8))
                    brake = self.brake_intensity
                    
            except KeyError as e:
                labels.append(f"{l} :: {round(s, 2)} :: {d}")
                
        return labels, brake
    
    def forward(self, image):
        x = [self.preprocess(F.to_pil_image(image)).to("cuda")]
        
        with torch.no_grad():
            predictions = self.model(x)
            
        
        prediction = self._remove_boxes(predictions[0])
        
        labels, brake = self._create_labels_and_brake(prediction)
        
        return prediction, labels, brake
        

 
#new_screen = draw_bounding_boxes(torch.tensor(screen).permute(2,0,1), predictions[0]["boxes"], labels).numpy()



#cv2.imshow('win1', canny_image)
#cv2.imshow('win1',cv2.cvtColor(np.moveaxis(new_screen, 0, 2), cv2.COLOR_BGR2RGB))
