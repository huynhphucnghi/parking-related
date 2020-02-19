import numpy as np 
import tensorflow as tf 
import os
import cv2
from model_01.model import TiniYOLO
from dataset import DataGenerator
from loss import tini_yolo_loss
from train_test_split import TRAIN_DATA_SUMMARY, TEST_DATA_SUMMARY
from config import (
    INNIT_LEARNING_RATE,
    EXPONENENTIAL_RATE, 
    DECAY_STEP
    )

MODEL_CHECK_POINT_PATH = 'model_01/check_point/'
LOG_PATH = MODEL_CHECK_POINT_PATH + "log.txt"

# def exp_learning_rate_decay(innit_lr, step, decay_step=DECAY_STEP, rate=EXPONENENTIAL_RATE):
#     return innit_lr * np.power(rate, int(step/decay_step))

def write_log(data, log_path):
    with open(log_path, "a") as myfile:
        data = [str(x) for x in data]
        myfile.write('\t'.join(data) + '\n')

if __name__ == "__main__":

    model = TiniYOLO()
    optimizer = model.optimizer()
    model = model.model(training=True)
    model.compile(optimizer=optimizer,
              loss=tini_yolo_loss)
    model.load_weights( MODEL_CHECK_POINT_PATH + 'cp{}.ckpt'.format(80))
    train_dataset = DataGenerator(TRAIN_DATA_SUMMARY)
    test_dataset = DataGenerator(TEST_DATA_SUMMARY)
    steps_per_epoch = train_dataset.steps_per_epoch
    learning_rate = 0.000001

    # with open(LOG_PATH, 'w') as wf:
    #     wf.write('{}\t{}\t{}\n'.format('step', 'train_loss', 'test_loss'))

    for i in range(80, 150):
        train_data = train_dataset.generator(training=True)
        test_data = test_dataset.generator(training=False)
        print("step {}".format(i + 1))
        
        model.optimizer.lr = learning_rate
        print(model.optimizer.lr)
        history = model.fit_generator(train_data, steps_per_epoch=steps_per_epoch, epochs=1)

        train_loss = (history.history['loss'])[0]
        test_loss = []
        for x_test, y_test in test_data:
            loss = model.evaluate(x_test, y_test, verbose=False)
            test_loss.append(loss)
        test_loss = np.mean(test_loss)
        print("step {}\ttrain loss {}\ttest loss {}".format(i + 1, train_loss, test_loss))
        write_log([i, train_loss, test_loss], LOG_PATH)

        if ((i+1) % 10) == 0 :
            checkpoint_path = MODEL_CHECK_POINT_PATH + "cp{}.ckpt".format(i + 1)
            model.save_weights(checkpoint_path)
            # learning_rate = 0.001 / (10 * int(i / 10)) 
            if (i+1) >= 45:
                learning_rate = 0.00005
            if (i+1) >= 55:
                learning_rate = 0.000001
