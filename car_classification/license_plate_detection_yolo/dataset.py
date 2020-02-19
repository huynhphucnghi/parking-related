import numpy as np 
import cv2
import random
from config import (
    BATCH_SIZE, 
    IMAGE_SIZE, 
    GRID_SIZE
)
from utils import draw_bbox, show_img
from train_test_split import TRAIN_DATA_SUMMARY, TEST_DATA_SUMMARY
def read_summary(summary):
    """
        get all information from summary.txt
    """
    content = []
    with open(summary, 'r') as sm:
        for line in sm:
            content.append(line.strip())
    
    return content

def transform_img(img, img_size):
    """
        read image and preprocess by dividing 255
    """
    img = cv2.resize(img, (img_size, img_size))
    img = img / 255
    return np.array(img)
    # return img

def transform_box(box, original_shape, grid_size, has_license):
    """
        need to change it here
        note that in this dataset
        box is construct with type (min_x, min_y, width, height) 
        x is horizontal axis
        y is vertical axis
        the origin is on the top-left corner
    """
    one_hot = np.zeros((grid_size, grid_size, 5))
    if has_license == False:
        return one_hot
    # dm open cv, shape is (h, w, chanel)
    true_h, true_w, _ = original_shape
    box = [int(x) for x in box]
    box_x, box_y, box_w, box_h = box[0], box[1], (box[2] - box[0]), (box[3] - box[1])
    x1, x2 = box_x, (box_x + box_w)
    y1, y2 = box_y, (box_y + box_h)
    
    grid_w = true_w / grid_size
    grid_h = true_h / grid_size
    # devide image into grid_size^2 part, find if the center of the license plate is in which part
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2

    pos_x = np.floor(x_center / grid_w)
    pos_y = np.floor(y_center / grid_h)
    # print(x_center, y_center)
    # print(pos_x, pos_y)

    x_center = (x_center - pos_x * grid_w) / grid_w
    y_center = (y_center - pos_y * grid_h) / grid_h
    w = box_w / grid_w
    h = box_h / grid_h
    
    
    index_x = int(pos_x)
    index_y = int(pos_y)
    one_hot[index_y, index_x, 0] = x_center 
    one_hot[index_y, index_x, 1] = y_center
    one_hot[index_y, index_x, 2] = w
    one_hot[index_y, index_x, 3] = h
    one_hot[index_y, index_x, 4] = 1.0
    # print(one_hot)
    return one_hot

def decode_bounding_box(box, shape, grid_size, pro_thresh_hold=0.4):
    """
        box is (x_center, y_center, w, h)
        this function is convert it to (x1, y1, x2, y2)
    """

    # x_center, y_center, w, h = box[0:4]
    # one_hot = box[4:]
    # position = np.argmax(one_hot)
    # pos_y = np.floor(position / grid_size)
    # pos_x = position - pos_y * grid_size
    position = np.argmax(box[..., 4])
    pos_y = np.floor(position / grid_size)
    pos_x = position - pos_y * grid_size
    if box[int(pos_y), int(pos_x), 4] < pro_thresh_hold:
        return (0, 0, 0, 0)
    x_center, y_center, w, h = box[int(pos_y), int(pos_x), 0:4]
    # print(position)
    # print(box[..., 4])
    # print(box)
    # print(x_center, y_center, w, h)
    true_w, true_h, _ = shape
    
    grid_w = true_w / grid_size
    grid_h = true_h / grid_size

    x_center = x_center*grid_w + pos_x*grid_w
    y_center = y_center*grid_h + pos_y*grid_h
    w = w * grid_w
    h = h * grid_h 

    x1 = int(x_center - w/2)
    x2 = int(x_center + w/2)
    y1 = int(y_center - h/2)
    y2 = int(y_center + h/2)
    return (x1, y1, x2, y2)

def get_batch_description(data, batch_size):
    """
        get list of data with shape (number of data samples) 
        shuffle it and separated it with batch size 
        it is easy to ignore some leftover samples at the end
    """
    size = np.shape(data)[0] # take shape of data
    result = []
    for i in range(int(np.floor(size/batch_size))):
        result.append(data[i*batch_size : (i+1)*batch_size])
    return result

def load_data(data, img_size):
    x, y, z = [], [], []
    for sample in data:
        sample = sample.strip().split()
        img = cv2.imread(sample[0])
        img_shape = img.shape
        x.append(transform_img(img, img_size))
        z.append(img)
        if len(sample) == 1:
            y.append(transform_box(None, None, GRID_SIZE, has_license=False))
        else :
            box = sample[1:5]
            y.append(transform_box(box, img_shape, GRID_SIZE, has_license=True))
            
    return np.array(x), np.array(y), np.array(z)

class DataGenerator(object):
    def __init__(self,
                summary,
                batch_size=BATCH_SIZE,
                img_size = IMAGE_SIZE
                ):
        self.batch_size = batch_size
        self.img_size = img_size
        self.data = read_summary(summary)
        self.steps_per_epoch = np.floor(len(self.data) / self.batch_size)

    def generator(self, training):
        if training == True:
            random.shuffle(self.data)
        batchs = get_batch_description(self.data, self.batch_size)
        print(np.shape(batchs))
        for batch in batchs:
            # print(batch)
            x, y, z = load_data(batch, self.img_size)
            yield x, y


if __name__ == '__main__':
    x = DataGenerator(TRAIN_DATA_SUMMARY, img_size=1024).generator(training=True)
    
    # show_img(cv2.imread('data/license/21744.jpg'))
    for j in range(20):
        a, b= next(x)
        for i in range(len(b)):
            img = a[i]
            img_shape = np.shape(img)
            box = decode_bounding_box(b[i], img_shape, GRID_SIZE)
            img = draw_bbox(img, box)
            show_img(img, str(img_shape))
