import cv2

def draw_bbox(img, box, color=(255, 0, 0)):
    box = [int(x) for x in box]
    return cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color)

def show_img(img, title='Fuck You, Draw more'):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def cut_img_bbox(img, box):
    box = [int(x) for x in box]
    img = img[box[1]:(box[3]+1), box[0]:(box[2]+1), :]
    return img

def save_img(img, img_url):
    cv2.imwrite(img_url, img)

def rgb2gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def resize_img(img, size):
    return cv2.resize(img, size) 