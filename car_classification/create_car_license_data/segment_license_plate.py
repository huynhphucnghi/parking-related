import numpy as np 
import cv2
import os

from utils import draw_bbox, show_img, cut_img_bbox, save_img
SUMMARY = 'data/summary.txt'
OUTPUT_URL = 'segmented_data'
def read_summary(summary):
    """
        get all information from summary.txt
    """
    content = []
    with open(summary, 'r') as sm:
        for line in sm:
            content.append(line.strip())
    
    return content

def transform_box(box):
    """
        need to change it here
        note that in this dataset
        box is construct with type (min_x, min_y, width, height) 
        x is horizontal axis
        y is vertical axis
        the origin is on the top-left corner
    """
    # dm open cv, shape is (h, w, chanel)
    box = [int(x) for x in box]
    x1, y1, x2, y2 = box
    return (x1, y1, x2, y2)

def get_bounding_box(data):
    """
        Using data form read_summary to create a bounding box of license plate
    """
    for row in data:
        row = row.split()
        img_url = row[0]
        img_coord = row[1:5]
        img_coord = transform_box(img_coord)
        # ground_true = row[5]
        yield img_url, img_coord

def cut_bounding_box(img_url, img_coord, output_url):
    img = cv2.imread(img_url)
    img = cut_img_bbox(img, img_coord)
    img_name = img_url.split('/')[-1]
    img_name = os.path.join(output_url, img_name)
    print(img_name)
    # show_img(img)
    save_img(img, img_name)

if __name__=='__main__':
    data = read_summary(SUMMARY)
    print(data[0:10])
    hello_world = get_bounding_box(data)
    for row in hello_world:
        img_url, img_coord = row
        cut_bounding_box(img_url, img_coord, OUTPUT_URL)