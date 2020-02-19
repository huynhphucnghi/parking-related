import numpy as np
import random 
import cv2
from utils import show_img, resize_img
import config

from train_test_split import TEST_SUMMARY, TRAIN_SUMMARY


def read_summary_file(summary):
    """
        Input:
            summary : an .txt file name contain all information about dataset,
                it is created by summary.py with format: (path_to_file, label) in each line             
        Output:
            result : an array contain all information about dataset, with shape
                    (number_of_files, (path, label))
    """
    result = []
    with open(summary, 'r') as sm:
        for line in sm:
            path, label = line.strip().split(' ')
            label = int(label)
            result.append((path, label))

    return np.array(result)

def shuffle_dataset(alldata):
    """
        Input:
            alldata : an array contain all information about dataset, with shape
                    (number_of_filse, (path, label))
        Output:
            result: an array like alldata, but the order of data's samples is shuffled
    """
    np.random.shuffle(alldata)
    return alldata

def standardize_img(img_file_path, img_w, img_h):
    """
        Input: 
            img_file_path : this is path to image file
        Output:
            img : an three dimensions numpy array of the image above. It is standardized by
                resizing the image to particular shape (img_w, img_h) 
                and dividing by 255 to make value range (0, 1)
    """
    img = cv2.imread(img_file_path)
    img = resize_img(img, (img_w, img_h))
    # show_img(img)
    # print(np.shape(img))
    img = img / 255
    return np.array(img)

class DataGenerator(object):

    def __init__(self, 
                summary, 
                batch_size=config.BATCH_SIZE, 
                input_width=config.INPUT_WIDTH,
                input_height=config.INPUT_HEIGHT
                ):
        self.batch_size = batch_size
        self.input_width = input_width
        self.input_height = input_height

        self.train_data = read_summary_file(summary)

        self.number_of_training_batches = int(np.shape(self.train_data)[0] / self.batch_size)
        

    def gen_batch(self, data):
        """
            Input:
                data: a batch of information of file path and label 
                        with shape = (batch_size, (path, label))
            Output:
                x : a numpy array contains information of image
                y : a numpy array contains information of label
        """
        batch_size = np.shape(data)[0]
        x, y = [], []
        for i in range(batch_size):
            img, label = data[i]
            img = standardize_img(img, self.input_width, self.input_height)
            x.append(img)
            y.append(int(label))
            # show_img(img)
        return np.array(x), np.array(y)

    def data_generator(self, training=True):
        """
            Input:
                summary : an .txt file name contain all information about dataset,
                    it is created by summary.py with format: (path_to_file, label) in each line             
            Output:
                result : an iterator loop through dataset, make data follow batch
        """
        if training:
            self.train_data = shuffle_dataset(self.train_data)

        for i in range(self.number_of_training_batches):
            temp_batch = self.train_data[i*self.batch_size : (i+1)*self.batch_size]
            x_batch, y_batch = self.gen_batch(temp_batch)
            yield x_batch, y_batch

if __name__=='__main__':
    # a = DataGenerator(TRAIN_SUMMARY, test_summary=TEST_SUMMARY)
    # print(len(a.train_data), len(a.test_data))
    # train = a.train_generator()
    # x, y = next(train)
    # print(len(x), len(y))
    # print(y)
    # show_img(x[0])
    source = './data/yes/10148.jpg'
    img = standardize_img(source, 100, 30)
    show_img(img)