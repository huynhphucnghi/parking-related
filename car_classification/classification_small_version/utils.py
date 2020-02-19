import cv2
import numpy as np

def show_img(img, title='Fuck You, Draw more'):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def resize_img(img, size):
    img_w, img_h = size
    img = cv2.resize(img, (img_w, img_h))
    return img

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