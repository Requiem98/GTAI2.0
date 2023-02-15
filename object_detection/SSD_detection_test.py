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


def euclideanDistance(coordinates):
    return pow(pow(coordinates[0] - coordinates[2], 2) + pow(coordinates[1] - coordinates[3], 2), .5)

classes = {3 : "car", 1 : "person", 8 : "truck", 4 : "cycle"}

model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)
model.to("cuda")
model.eval()
preprocess = SSD300_VGG16_Weights.DEFAULT.transforms()


left = int((2560 - 800)/2)
top = int((1440 - 600)/2)

cv2.namedWindow("win1");
<<<<<<< HEAD
cv2.moveWindow("win1", 75,top-40);
=======
cv2.moveWindow("win1", 0,0);
>>>>>>> 1e5b878eb15a18030a64a31772161c9a7ef6ed50

last_time = time.time()
while True:
    screen =  np.array(ImageGrab.grab(bbox=(left, top,left+800,top+600)))
    print('Frame took {} seconds'.format(time.time()-last_time))
    last_time = time.time()
    
    
    x = [preprocess(F.to_pil_image(screen)).to("cuda")]

    with torch.no_grad():
        predictions = model(x)

    
    mask = (predictions[0]['scores'] > 0.40) & (
        (predictions[0]['labels'] == 3) | (predictions[0]['labels'] == 1) | (predictions[0]['labels'] == 8) | (predictions[0]['labels'] == 4))
            
            
    mask = predictions[0]['scores'] > 0.40
    for key,val in predictions[0].items():
        predictions[0][key] = val[mask]
    
    
    labels = list()
    
    
    for l, s, b in zip(predictions[0]["labels"].cpu().numpy(), predictions[0]["scores"].cpu().numpy(), predictions[0]["boxes"].cpu().numpy()):
        d = euclideanDistance(b)
        try:
            labels.append(f"{classes[l]} :: {round(s, 2)} :: {d}")
            
            if(d > 400):
                print("\n\nBREAK!\n\n")
    
        except KeyError as e:
            labels.append(f"{l} :: {round(s, 2)} :: {d}")
    
    
<<<<<<< HEAD
    new_screen = draw_bounding_boxes(torch.tensor(screen).permute(2,0,1), predictions[0]["boxes"], labels, colors = "red", width = 4).numpy()
=======
    new_screen = draw_bounding_boxes(torch.tensor(screen).permute(2,0,1), predictions[0]["boxes"], labels).numpy()
>>>>>>> 1e5b878eb15a18030a64a31772161c9a7ef6ed50
    

    
    #cv2.imshow('win1', canny_image)
    cv2.imshow('win1',cv2.cvtColor(np.moveaxis(new_screen, 0, 2), cv2.COLOR_BGR2RGB))

    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break