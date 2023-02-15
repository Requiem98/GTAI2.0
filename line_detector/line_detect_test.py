import numpy as np
from PIL import ImageGrab
import cv2
import time
import pyautogui
from numpy import ones,vstack
from numpy.linalg import lstsq
from statistics import mean
from libraries import *


from sklearn.linear_model import HuberRegressor, Ridge



class LineDetector:
    def __init__(self, line_threshold = 330, line_thickness = 6):
        self.line_threshold = line_threshold
        self.line_thickness = line_thickness
        
    
    def roi(self, img, vertices):
        
        #blank mask:
        mask = np.zeros_like(img)   
        
        #filling pixels inside the polygon defined by "vertices" with the fill color    
        cv2.fillPoly(mask, vertices, 255)
        
        #returning the image only where mask pixels are nonzero
        masked = cv2.bitwise_and(img, mask)
        return masked


    def draw_lines(self, img, lines, mask):
        line_dict = {'left':[], 'right':[]}
        img_center = img.shape[1]//2
        
        for line in lines:
            for x1, y1, x2, y2 in line:
                if x1<img_center and x2<img_center:
                    position = 'left'
                elif x1>img_center and x2>img_center:
                    position = 'right'
                else:
                    continue
                line_dict[position].append(np.array([x1, y1]))
                line_dict[position].append(np.array([x2, y2]))

        for position, lines_dir in line_dict.items():
            data = np.array(lines_dir)
            data = data[data[:, 1] >= np.array(self.line_threshold)-1]
            x, y = data[:, 0].reshape((-1, 1)), data[:, 1]

            try:
                model = HuberRegressor(fit_intercept=True, alpha=0.0, max_iter=100,
                                    epsilon=1.9)
                model.fit(x, y)
            except ValueError:
                model = Ridge(fit_intercept=True, alpha=0.0, random_state=0, normalize=True)
                model.fit(x, y)

            epsilon = 1e-10
            y1 = np.array(img.shape[0])
            x1 = (y1 - model.intercept_)/(model.coef_+epsilon)
            y2 = np.array(self.line_threshold)
            x2 = (y2 - model.intercept_)/(model.coef_+epsilon)
            x = np.append(x, [x1, x2], axis=0)
            
            y_pred = model.predict(x)
            data = np.append(x, y_pred.reshape((-1, 1)), axis=1)
            
            cv2.polylines(img, np.int32([data]), isClosed=False,
                          color=(255, 0, 0), thickness=self.line_thickness)
            
            cv2.polylines(mask, np.int32([data]), isClosed=False,
                          color=(255, 0, 0), thickness=self.line_thickness)
            
        



    def process_img(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        processed_img = cv2.GaussianBlur(gray_image,(5,5),0)
        
        # edge detection
        processed_img =  cv2.Canny(processed_img, threshold1 = 200, threshold2=300)
        
        vertices = np.array([[0,480], [800, 480], [800, 380], [600, 300], [200, 300], [0, 380]], np.int32)

        processed_img = self.roi(processed_img, [vertices])
        

        # more info: http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html
        #                                     rho   theta   thresh  min length, max gap:        
        
        lines = cv2.HoughLinesP(processed_img, 1, np.pi/180, threshold=10, minLineLength=30, maxLineGap=40)
        
        mask = np.zeros_like(processed_img)
        
        try:
            self.draw_lines(image, lines, mask)
            
        except Exception as e:
            print(str(e))
            lines_coord = list()
          
        

        return image, processed_img, mask



    
    


"""
for i in list(range(4))[::-1]:
    print(i+1)
    time.sleep(1)
"""

line_finder = LineDetector()

left = int((2560 - 800)/2)
top = int((1440 - 600)/2)

cv2.namedWindow("win1");
<<<<<<< HEAD
cv2.moveWindow("win1", 75,top-40);

cv2.namedWindow("win2");
cv2.moveWindow("win2", left+795,top-40);
=======
cv2.moveWindow("win1", 0,0);
>>>>>>> 1e5b878eb15a18030a64a31772161c9a7ef6ed50

last_time = time.time()
while True:
    screen =  np.array(ImageGrab.grab(bbox=(left, top,left+800,top+600)))
    print('Frame took {} seconds'.format(time.time()-last_time))
    last_time = time.time()
    
    new_screen, canny_image, mask = line_finder.process_img(screen)

    cv2.line(new_screen, (300, 600), (300, 480), [0,0,255], 6)
    cv2.line(new_screen, (500, 600), (500, 480), [0,0,255], 6)
    
<<<<<<< HEAD
    #check_intersection(mask)

    
    cv2.imshow('win2', canny_image)
=======
    check_intersection(mask)

    
    #cv2.imshow('win1', canny_image)
>>>>>>> 1e5b878eb15a18030a64a31772161c9a7ef6ed50
    cv2.imshow('win1',cv2.cvtColor(new_screen, cv2.COLOR_BGR2RGB))

    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break