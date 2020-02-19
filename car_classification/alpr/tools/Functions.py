import math
import cv2


class Contour:
    # this function contains some operations used by various function in the code
    def __init__(self, cntr):
        self.contour = cntr

        self.boundingRect = cv2.boundingRect(self.contour)

        [x, y, w, h] = self.boundingRect

        self.boundingRectX = x
        self.boundingRectY = y
        self.boundingRectWidth = w
        self.boundingRectHeight = h

        self.boundingRectArea = self.boundingRectWidth * self.boundingRectHeight

        self.centerX = (self.boundingRectX + self.boundingRectX + self.boundingRectWidth) / 2
        self.centerY = (self.boundingRectY + self.boundingRectY + self.boundingRectHeight) / 2

        self.diagonalSize = math.sqrt((self.boundingRectWidth ** 2) + (self.boundingRectHeight ** 2))

        self.aspectRatio = float(self.boundingRectWidth) / float(self.boundingRectHeight)


def cut_img_bbox(img, box):
    box = [int(x) for x in box]
    img = img[box[1]:(box[3]+1), box[0]:(box[2]+1), :]
    return img

class Plate:

    def __init__(self, contours):
        self.contours = contours
        self.Plate = None
        self.Grayscale = None
        self.Thresh = None
        self.rrLocationOfPlateInScene = None
        self.strChars = ""
        self.box = None # box is (x1, y1, x2, y2)
        self.area = None
        self.width = None
        self.height = None
        self.center_x = None 
        self.center_y = None
    
    def get_extend_box(self):
        "this funtion is get around area of very tight box, make it to perform detection"
        extend_width = int(self.width * 0.15)
        extend_height = int(self.height * 0.3)
        x1, y1, x2, y2 = self.box
        e_x1 = x1 - extend_width
        e_x2 = x2 + extend_width
        e_y1 = y1 - extend_height
        e_y2 = y2 + extend_height
        return (e_x1, e_y1, e_x2, e_y2)

    def get_plate_img(self, img):
        box = self.get_extend_box()
        x1, y1, x2, y2 = box 
        h, w, c = img.shape

        if x1 < 0: x1 = 0
        if y1 < 0: y1 = 0
        if x2 > w: x2 = w 
        if y2 > h: y2 = h
        box = (x1, y1, x2, y2)
        return cut_img_bbox(img, box)

# this function is a 'first pass' that does a rough check on a contour to see if it could be a char
def checkIfChar(possibleChar):
    if (0.2 < possibleChar.aspectRatio < 1.5):
        return True

    return False

# check the center distance between characters
def distanceBetweenChars(firstChar, secondChar):
    x = abs(firstChar.centerX - secondChar.centerX)
    y = abs(firstChar.centerY - secondChar.centerY)

    return math.sqrt((x ** 2) + (y ** 2))

def distanceBetweenPlates(plate_one, plate_two):
    x = abs(plate_one.center_x - plate_two.center_x)
    y = abs(plate_one.center_y - plate_two.center_y)

    return math.sqrt((x ** 2) + (y ** 2))

# use basic trigonometry (SOH CAH TOA) to calculate angle between chars
def angleBetweenChars(firstChar, secondChar):
    adjacent = float(abs(firstChar.centerX - secondChar.centerX))
    opposite = float(abs(firstChar.centerY - secondChar.centerY))

    # check to make sure we do not divide by zero if the center X positions are equal
    # float division by zero will cause a crash in Python
    if adjacent != 0.0:
        angleInRad = math.atan(opposite / adjacent)
    else:
        angleInRad = 1.5708

    # calculate angle in degrees
    angleInDeg = angleInRad * (180.0 / math.pi)

    return angleInDeg
