from libraries import *
from torchvision.utils import draw_bounding_boxes
import numpy as np
from PIL import ImageGrab
import cv2
import time
import pyautogui
from numpy import ones,vstack
from numpy.linalg import lstsq
from statistics import mean


def euclideanDistance(coordinates):
    return pow(pow(coordinates[0] - coordinates[2], 2) + pow(coordinates[1] - coordinates[3], 2), .5)

classes = {3 : "car", 1 : "person", 8 : "truck", 4 : "cycle"}

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.cuda()


left = int((2560 - 800)/2)
top = int((1440 - 600)/2)

cv2.namedWindow("win1");
cv2.moveWindow("win1", 75,top-40);

last_time = time.time()
while True:
    screen =  np.array(ImageGrab.grab(bbox=(left, top,left+800,top+600)))
    print('Frame took {} seconds'.format(time.time()-last_time))
    last_time = time.time()
    

  
    results = model(screen)
    results.render()
    results.crop()  #(xyxy, conf, cls)
    results.xyxy[0][:, :4]
    
    #new_screen = draw_bounding_boxes(torch.tensor(screen).permute(2,0,1), predictions[0]["boxes"], labels, colors = "red", width = 4).numpy()
    
    cv2.imshow('win1',cv2.cvtColor(results.ims[0], cv2.COLOR_BGR2RGB))

    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break