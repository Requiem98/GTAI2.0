import time
import csv
import win32gui, win32api
import os
import sys
import PIL
from PIL import Image
import mss
import pygame

path_name = "path_traffic_4/"

images_save_path = "./Data/gta_data/" + path_name + "images/"
csv_save_path = "./Data/gta_data/" + path_name

samples_per_second=30

if not os.path.exists(images_save_path):
    os.makedirs(images_save_path)

csv_file = open(csv_save_path + 'data.csv', 'w+')

csv_file.write('steering_angle,throttle,brake,speed,path\n')

print('Recording starts in 5 seconds...')
time.sleep(5)
print('Recording started!')

current_sample=0
last_time=0
start_time=time.time()
wait_time=(1/samples_per_second)
stats_frame=0


sct = mss.mss()

left = int((2560 - 800)/2)
top = int((1440 - 600)/2)

mon = {'top': top, 'left': left, 'width': 800, 'height': 600}

pause=False
return_was_down=False


pygame.display.init()
pygame.joystick.init()
joysticks = [pygame.joystick.Joystick(x) for x in range(pygame.joystick.get_count())]
joysticks[0].init()

while True:
    pygame.event.pump()
	
    if (win32api.GetAsyncKeyState(0x08)&0x8001 > 0):
        print("STOP!")
        break

    if (win32api.GetAsyncKeyState(0x0D)&0x8001 > 0):
        if (return_was_down == False):
            if (pause == False):
            
                pause = True
            else:
                pause = False
				
        return_was_down = True
    else:
        return_was_down = False

    if (time.time() - last_time >= wait_time):
	
        fps=1 / (time.time() - last_time)
        last_time = time.time()
		
        stats_frame+=1
        if (stats_frame >= 10):
            stats_frame=0
            os.system('cls')
            print('FPS: %.2f Total Samples: %d Time: %s' % (fps, current_sample, time.strftime("%H:%M:%S",time.gmtime(time.time() - start_time))))
            if (pause == False):
                print('Status: Recording')
            else:
                print('Status: Paused')
		
		
        if (pause):
            time.sleep(0.01)
            continue
        
        file = open("./Data/speed.txt", "r")
        new_speed = file.read()
        file.close()
		
        if (len(new_speed) > 0):
            speed = float(new_speed)
		
        if(speed > 0):
            sct_img = sct.grab(mon)
            image = Image.frombytes('RGB', sct_img.size, sct_img.rgb)
            


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




                        
            path = images_save_path + 'img%d.jpg' % current_sample
            image.save(path)
            csv_file.write('%f,%f,%f,%f,%s\n' % (steering_angle, throttle, brake, speed, path))
            
            
                
            current_sample += 1
		
	
	
print('\nDONE')
print('Total Samples: %d\n' % current_sample)

joysticks[0].quit()
pygame.display.quit()
pygame.joystick.quit()