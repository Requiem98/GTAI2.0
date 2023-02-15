from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import sys
import mss
import win32gui, win32api
import time
import PIL
import pyvjoy
import cv2
from libraries import *
import baseFunctions as bf

from simulator.directkeys import PressKey
from simulator.directkeys import ReleaseKey
from PIL import Image
from line_detector.LineDetector import *
<<<<<<< HEAD
from object_detection.YOLO import *

from Models.MapNet_v8 import *
CKP_DIR = "./Data/Models/MapNet_v8/checkpoint/"

left = int((2560 - 800)/2)
top = int((1440 - 600)/2)

#cv2.namedWindow("win1");
#cv2.moveWindow("win1", 75,top-40);

cv2.namedWindow("win1");
cv2.moveWindow("win1", left+795,top-40);
=======
from object_detection.SSD import *

from models.ViT_MapNet_v1 import *
CKP_DIR = "./Data/models/ViT_MapNet_v1/checkpoint/"

cv2.namedWindow("win1");
cv2.moveWindow("win1", 0,0);
>>>>>>> 1e5b878eb15a18030a64a31772161c9a7ef6ed50

j = pyvjoy.VJoyDevice(1)

vjoy_max = 32768

preprocess = bf.PREPROCESS()

def get_model(device):
<<<<<<< HEAD
    model = MapNet_v8().to(device) #qui inserire modello da trainare
    model.load_state_dict(torch.load(CKP_DIR+ "00050.pth"))
=======
    model = ViT_MapNet_v1(device = device).to(device) #qui inserire modello da trainare
    model.load_state_dict(torch.load(CKP_DIR+ "00150.pth"))
>>>>>>> 1e5b878eb15a18030a64a31772161c9a7ef6ed50
    return model



		
def predict_loop(device, model):
    
<<<<<<< HEAD
    obj_model = YOLO(device, conf = 0.3, min_distance = (300, 150), brake_intensity = 1.0)
=======
    ssd_model = SSD(min_distance = (300, 150), brake_intensity = 1.0)
>>>>>>> 1e5b878eb15a18030a64a31772161c9a7ef6ed50
    line_detector = LineDetector()
    
    model.eval()
    
	
    pause=True
    return_was_down=False
	

    sct = mss.mss()
    
    left = int((2560 - 800)/2)
    top = int((1440 - 600)/2)
    
    mon = {'top': top, 'left': left, 'width': 800, 'height': 600}
	
    i=0
	
    
    print('Ready')
	
    
    while True:
        i += 1
        if (win32api.GetAsyncKeyState(0x08)&0x8001 > 0):
            print("stop!")
            break
		
        if (win32api.GetAsyncKeyState(0x0D)&0x8001 > 0):
            if (return_was_down == False):
                if (pause == False):
                    pause = True
					
                    j.data.wAxisX = int(vjoy_max * 0.5)
                    j.data.wAxisY = int(vjoy_max * 0)
                    j.data.wAxisZ = int(vjoy_max * 0)

                    j.update()
					
                    print('Paused')
                else:
                    pause = False
					
                    print('Resumed')
				
            return_was_down = True
        else:
            return_was_down = False
		
        if (pause):
            time.sleep(0.01)
            continue
        

        file = open("./Data/speed.txt", "r")
        new_speed = file.read()
        file.close()
		
        if (len(new_speed) > 0):
            speed = float(new_speed)/35.5
        else:
            speed = 0.0

        sct_img = sct.grab(mon)
        orig_image = np.array(Image.frombytes('RGB', sct_img.size, sct_img.rgb))
        
        mmap = preprocess.preprocess_mmap_predict(orig_image)        
        image = preprocess.preprocess_image_predict(orig_image)
           
<<<<<<< HEAD
        #pred = model(image.to(device), mmap.to(device))
        pred = model(image.to(device), mmap.to(device), torch.tensor([speed]).to(device).unsqueeze(1))

        steeringAngle = pred.cpu().detach().numpy()[0].item()

        brake = 0.0
        
        
        if(speed >= 0.5):
=======
		
        pred1, pred2 = model(image.to(device), mmap.to(device), torch.tensor([speed]).to(device).unsqueeze(1))
        
        pred_brk = (torch.sigmoid(pred2.flatten()) > 0.5).to(dtype=torch.int32)

        steeringAngle = pred1[:, 0].cpu().detach().numpy()[0].item()

        brake = 0.0

        #pred = model(image.to(device), mmap.to(device))
        
        
        #steeringAngle = pred[:, 0].cpu().detach().numpy()[0].item()
        #throttle = pred[:, 1].cpu().detach().numpy()[0].item()
        #brake = pred[:, 2].cpu().detach().numpy()[0].item()
        
        if(speed >= 0.3):
>>>>>>> 1e5b878eb15a18030a64a31772161c9a7ef6ed50
            throttle = 0.0
        elif(speed < 0.3 and speed >= 0.2):
            throttle = 0.3
        elif(speed < 0.2 and speed >= 0.1):
            throttle = 0.6
        elif(speed < 0.1):
            throttle = 0.8
<<<<<<< HEAD
        
        """
        if(speed >= 0.8):
            throttle = 0.0
        elif(speed < 0.8 and speed >= 0.4):
            throttle = 0.3
        elif(speed < 0.4 and speed >= 0.3):
            throttle = 0.6
        elif(speed < 0.3 and speed >= 0.2):
            throttle = 0.7
        elif(speed < 0.2 and speed >= 0.1):
            throttle = 0.8
        elif(speed < 0.1):
            throttle = 0.9
        """  
=======
            
>>>>>>> 1e5b878eb15a18030a64a31772161c9a7ef6ed50
        
        #new_screen, _, mask = line_detector.process_img(orig_image)
        
        #steeringAngle = check_intersection(mask)
        
<<<<<<< HEAD
        prediction, brake = obj_model.forward(orig_image)
        
        cv2.imshow('win1',cv2.cvtColor(prediction, cv2.COLOR_BGR2RGB))
        
        #new_screen = draw_bounding_boxes(torch.tensor(orig_image).permute(2,0,1), prediction["boxes"], labels, colors = "red", width = 4).numpy()
        
        #cv2.imshow('win1',cv2.cvtColor(np.moveaxis(new_screen, 0, 2), cv2.COLOR_BGR2RGB))
        
=======
        prediction, labels, brake = ssd_model.forward(orig_image)
        
        new_screen = draw_bounding_boxes(torch.tensor(orig_image).permute(2,0,1), prediction["boxes"], labels).numpy()
        
        cv2.imshow('win1',cv2.cvtColor(np.moveaxis(new_screen, 0, 2), cv2.COLOR_BGR2RGB))

>>>>>>> 1e5b878eb15a18030a64a31772161c9a7ef6ed50
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        
        j.data.wAxisX = int(vjoy_max * min(max(steeringAngle, 0), 1))
        j.data.wAxisY = int(vjoy_max * min(max(throttle, 0), 1))
        j.data.wAxisZ = int(vjoy_max * min(max(brake, 0), 1))
		
        j.update()
		
        os.system('cls')
        print("Steering Angle: %.2f" % min(max(steeringAngle, 0), 1))
        print("Throttle: %.2f" % min(max(throttle, 0), 1))
        print("Brake: %.2f" % min(max(brake, 0), 1))
        print("Speed: %.2f" % speed)
		
		
        

if __name__ == "__main__":
    if not torch.cuda.is_available():
        device=torch.device("cpu")
        print("Current device:", device)
    else:
        device=torch.device("cuda")
        print("Current device:", device, "- Type:", torch.cuda.get_device_name(0))
        bf.get_memory()
        
    model = get_model(device)
    predict_loop(device, model)
    