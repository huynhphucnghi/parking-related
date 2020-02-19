import numpy as np
import random 
import cv2
from utils import show_img, resize_img
import config
from train_test_split import TRAIN_SUMMARY_PATH, TEST_SUMMARY_PATH
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
            path, label = line.strip().split()
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
    img = img / 255
    return np.array(img)

class DataGenerator(object):

    def __init__(self, 
                summary, 
                batch_size=config.BATCH_SIZE, 
                input_size=config.INPUT_SIZE
                ):
        self.batch_size = batch_size
        self.input_size = input_size
        self.alldata = read_summary_file(summary)
        self.number_of_batches = int(np.shape(self.alldata)[0] / self.batch_size)
        self.number_per_step = 5


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
            img = standardize_img(img, self.input_size, self.input_size)
            x.append(img)
            y.append(int(label))
        return np.array(x), np.array(y)

    def data_generator(self, training=False, img_augmentation=False):
        """
            Input:
                summary : an .txt file name contain all information about dataset,
                    it is created by summary.py with format: (path_to_file, label) in each line             
            Output:
                result : an iterator loop through dataset, make data follow batch
        """
        if training:
            self.alldata = shuffle_dataset(self.alldata)
            
        for i in range(self.number_of_batches):
            temp_batch = self.alldata[i*self.batch_size : (i+1)*self.batch_size]
            x_batch, y_batch = self.gen_batch(temp_batch)
            if augmentation:
                x_batch, y_batch = augmentation(x_batch, y_batch)
            yield x_batch, y_batch

def augmentation(x_train, y_train):
    horizontal_flip = np.random.randint(0, 2)
    if horizontal_flip == 1:
        flip = True
    else:
        flip = False
    datagen = ImageDataGenerator(
                        rotation_range=20,
                        width_shift_range=0.2,
                        height_shift_range=0.2,
                        horizontal_flip=flip,
                        shear_range=0.1,
                        zoom_range=[0.9, 1.25]
                        )
    nat_cmnr = datagen.flow(x_train, y_train, batch_size=len(x_train))
    x_batch, y_batch = next(nat_cmnr) 

    return np.array(x_batch), np.array(y_batch)

if __name__=='__main__':
    dataset = DataGenerator(TEST_SUMMARY_PATH)
    

    for i in range(1):
        gen = dataset.data_generator(training=True, img_augmentation=True)
        x_batch, y_batch = next(gen)

        print(np.shape(x_batch))
        for i in range(np.shape(y_batch)[0]):
            show_img(x_batch[i], title=str(y_batch[i]))
    
