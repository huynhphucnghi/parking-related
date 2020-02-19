import numpy as np 
import tensorflow as tf 
import os
import cv2 

from model import LPModel
import dataset
from utils import draw_bbox, show_img

Weights = '.\\checkpoint\\cp6.ckpt'
sample_data = '.\\data\\br\\JRK5336.jpg'

if __name__=='__main__':
    model = LPModel()
    model.load_weights(Weights)

    origin_img = cv2.imread(sample_data)
    img_h, img_w, c = origin_img.shape
    input_img = np.expand_dims(dataset.transform_img(origin_img) ,0)

    out = model.model.predict(input_img)[0]
    x, y, w, h = out
    print(out)
    x1, x2 = int((x-w/2)*416), int((x+w/2)*416)
    y1, y2 = int((y-h/2)*416), int((y+h/2)*416)
    
    img = draw_bbox(input_img[0], (x1, y1, x2, y2))
    show_img(img)
