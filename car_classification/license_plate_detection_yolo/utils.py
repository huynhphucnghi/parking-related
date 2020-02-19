import cv2

def draw_bbox(img, box, color=(0, 0, 255)):
    box = [int(x) for x in box]
    return cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color)

def show_img(img, title='Fuck You, Draw more'):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    