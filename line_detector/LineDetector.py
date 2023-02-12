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





# Return true if line segments AB and CD intersect
def is_intersect(A,B,C,D):
    
    def ccw(A,B,C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
    
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)


def check_intersection(mask):
    command = 0.5
    
    mask_left = np.zeros_like(mask)
    mask_right = np.zeros_like(mask)

    cv2.line(mask_left, (300, 600), (300, 480), [1], 6)
    cv2.line(mask_right, (500, 600), (500, 480), [1], 6)
    
    out_left = cv2.bitwise_and(mask, mask_left)
    out_right = cv2.bitwise_and(mask, mask_right)
    
    if((out_left > 0).any()):
        command = 1.0
    elif((out_right > 0).any()):
        command = 0.0
    else:
        command = 0.5  
    
    return command
    
