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

from models.ViT_MapNet_v1 import *
CKP_DIR = "./Data/models/ViT_MapNet_v1/checkpoint/"


j = pyvjoy.VJoyDevice(1)

vjoy_max = 32768

preprocess = bf.PREPROCESS()

def get_model(device):
    model = ViT_MapNet_v1(device = device).to(device) #qui inserire modello da trainare
    model.load_state_dict(torch.load(CKP_DIR+ "00100.pth"))
    return model



class FPSTimer:
	def __init__(self):
		self.t = time.time()
		self.iter = 0
		
	def reset(self):
		self.t = time.time()
		self.iter = 0
		
	def on_frame(self):
		self.iter += 1
		if self.iter == 100:
			e = time.time()
			print('FPS: %0.2f' % (100.0 / (e - self.t)))
			self.t = time.time()
			self.iter = 0



def lerp(a, b, t):
	return (t * a) + ((1-t) * b)
		
def predict_loop(device, model):
    
    
    model.eval()
    
    timer = FPSTimer()
	
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
        image = np.array(Image.frombytes('RGB', sct_img.size, sct_img.rgb))

        mmap = preprocess.preprocess_mmap_predict(image)        
        image = preprocess.preprocess_image_predict(image)
        
           
		
        pred1, pred2 = model(image.to(device), mmap.to(device), torch.tensor([speed]).to(device).unsqueeze(1))
        
        pred_brk = (torch.sigmoid(pred2.flatten()) > 0.5).to(dtype=torch.int32)

        steeringAngle = pred1[:, 0].cpu().detach().numpy()[0].item()
        throttle = pred1[:, 1].cpu().detach().numpy()[0].item()
        brake = pred_brk.cpu().detach().numpy()[0].item()

        if(brake>0):
            brake = 0.8

        #pred = model(image.to(device), mmap.to(device))
        
        
        #steeringAngle = pred[:, 0].cpu().detach().numpy()[0].item()
        #throttle = pred[:, 1].cpu().detach().numpy()[0].item()
        #brake = pred[:, 2].cpu().detach().numpy()[0].item()
		
        
        j.data.wAxisX = int(vjoy_max * min(max(steeringAngle, 0), 1))
        j.data.wAxisY = int(vjoy_max * min(max(throttle, 0), 1))
        j.data.wAxisZ = int(vjoy_max * min(max(brake, 0), 1))
		
        j.update()
		
        os.system('cls')
        print("Steering Angle: %.2f" % min(max(steeringAngle, 0), 1))
        print("Throttle: %.2f" % min(max(throttle, 0), 1))
        print("Brake: %.2f" % min(max(brake, 0), 1))
        print("Speed: %.2f" % speed)
		
        
		#timer.on_frame()
        

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
    