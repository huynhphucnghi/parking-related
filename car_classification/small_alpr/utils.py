import cv2
import numpy as np

def show_img(img, title='Haha'):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def resize_img(img, size):
    img_w, img_h = size
    img = cv2.resize(img, (img_w, img_h))
    return img

def draw_bbox(img, box, color=(0, 255, 0)):
    box = [int(x) for x in box]
    return cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color)

def cut_img_bbox(img, box):
    box = [int(x) for x in box]
    img = img[box[1]:(box[3]+1), box[0]:(box[2]+1), :]
    return img

def rgb2gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def get_conv_layer(model, name):
    """
        use to get a layer w, b from name of that layer
        if can not find that layer, throw an exception
    """
    for layer in model.layers:
        if layer.name == name:
            w, b = layer.get_weights()
            print('name: {}, weight_shape: {}, bias shape: {}'.format(name, np.shape(w), np.shape(b)))
            return w, b
    raise Exception('Can not find layer with name: {}'.format(name))


def get_bn_layer(model, name):
    """
        use to get batch normalization weight
    """
    for layer in model.layers:
        if layer.name == name:
            w = layer.get_weights()
            print('name: {}, weight: {}'.format(name, np.shape(w)))
            return w
    raise Exception('Can not find layer with name: {}'.format(name))

def get_dense_layer(model, name):
    """
        use to get dense weights
    """
    w, b = get_conv_layer(model, name)
    return w, b

def set_weights(model, name, weights):
    """
        use to set weight for model
    """
    for layer in model.layers:
        if layer.name == name:
            layer.set_weights(weights)
            return True
    raise Exception('Can not set weights with layer\'s name: {}'.format(name))