#!/bin/python

"""
USAGE:
    feature-point-detector-example.py <image>
"""

import cv2
import math
import numpy as np
import docopt
from scipy import signal

G = np.outer(signal.gaussian(5, 1), signal.gaussian(5, 1))

def derivative_of_gaussian(x):
    return -(math.exp(-math.pow(x,2)/2)*x)/math.sqrt(2*math.pi)
    
dGx = np.array([[derivative_of_gaussian(-1), derivative_of_gaussian(0), derivative_of_gaussian(1)],
                [derivative_of_gaussian(-1), derivative_of_gaussian(0), derivative_of_gaussian(1)],
                [derivative_of_gaussian(-1), derivative_of_gaussian(0), derivative_of_gaussian(1)]], dtype=float)

arguments = docopt.docopt(__doc__)
img_color = cv2.imread(arguments['<image>'])
img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
img = img.astype(float)

Ix = np.abs(cv2.filter2D(img, -1, dGx))
Iy = np.abs(cv2.filter2D(img, -1, np.transpose(dGx)))

Ix2 = Ix*Ix
Ixy = Ix*Iy
Iy2 = Iy*Iy

AIx2 = cv2.filter2D(Ix2, -1, G)
AIxy = cv2.filter2D(Ixy, -1, G)
AIy2 = cv2.filter2D(Iy2, -1, G)

for j in range(0, AIx2.shape[0]):
    for i in range(0, AIx2.shape[1]):
        A = np.array([[AIx2[j,i], AIxy[j,i]],[AIxy[j,i], AIy2[j,i]]])
        l = np.linalg.eig(A)[0]
        if(l[0] != 0 and l[1] != 0):
            l0 = min(l)
            if(l0 > 500):
                img_color[j:j+2,i:i+2] = [0,0,255]

cv2.imshow("Detected Key Points", img_color)
cv2.waitKey()
