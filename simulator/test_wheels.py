# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 21:24:56 2023

@author: amede
"""

import time
import csv
import win32gui, win32api
import os
import sys
import PIL
from PIL import Image
import mss
import pygame



pygame.display.init()
pygame.joystick.init()
joysticks = [pygame.joystick.Joystick(x) for x in range(pygame.joystick.get_count())]
joysticks[0].init()



while True:
    pygame.event.pump()
    
    
    steering_angle=joysticks[0].get_axis(0)
    if(abs(steering_angle) < 0.008):
        steering_angle=0.0
    if(steering_angle > 0.99):
        steering_angle=1.0
    if(steering_angle < -0.99):
        steering_angle=-1.0
            
    steering_angle=(steering_angle+1)/2
    
    
    throttle=joysticks[0].get_axis(1)
    
    if (throttle > 0.99):
        throttle=1
        
    throttle=1-(throttle+1)/2
    
    brake=joysticks[0].get_axis(2)
    if (brake > 0.99):
        brake=1
        
    brake=1-(brake+1)/2
    
    print(steering_angle, "  ", throttle, "  ", brake)
    #print(joysticks[0].get_axis(3))


joysticks[0].quit()
pygame.display.quit()
pygame.joystick.quit()