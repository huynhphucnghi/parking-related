import cv2
import glob
import numpy as np 
from utils import show_img, draw_bbox, rgb2gray, resize_img
from tools import Functions

def mean_adaptive_threshold(img):
	threshold = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
	return threshold

def gaussian_adaptive_threshold(img):
	threshold = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
	return threshold

def find_contour(img):
	_, contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	return contours

def preprocess(img):

	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	hue, saturation, value = cv2.split(hsv)
	value = cv2.equalizeHist(value)
	# kernel to use for morphological operations
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
	# applying topHat/blackHat operations
	topHat = cv2.morphologyEx(value, cv2.MORPH_TOPHAT, kernel)
	blackHat = cv2.morphologyEx(value, cv2.MORPH_BLACKHAT, kernel)

	# add and subtract between morphological operations
	add = cv2.add(value, topHat)
	subtract = cv2.subtract(add, blackHat)

	# applying gaussian blur on subtract image
	blur = cv2.GaussianBlur(subtract, (5, 5), 0)

	threshold = gaussian_adaptive_threshold(blur)
	contours = find_contour(threshold)
	return contours, threshold

def matching_chars(c, pos_char, check_list):
    """
        using to finding a list of character like c and can be a list of license numbers
    """
    list_matching_chars = [c]
    count = 0
    for pos_matching in pos_char:
        if (c == pos_matching) or (pos_matching in check_list):
            continue
        for char in list_matching_chars:
            
            distance_between_chars = Functions.distanceBetweenChars(char, pos_matching)

            angle_between_chars = Functions.angleBetweenChars(char, pos_matching)

            area_ratio = float(abs(char.boundingRectArea - pos_matching.boundingRectArea)) / float(
                    char.boundingRectArea)
            
            width_ratio = float(abs(char.boundingRectWidth - pos_matching.boundingRectWidth)) / float(
                    char.boundingRectWidth)
            
            height_ratio = float(char.boundingRectHeight - pos_matching.boundingRectHeight) / float(
                    char.boundingRectHeight)

            if distance_between_chars < (char.diagonalSize * 2) and \
                        angle_between_chars < 10.0 and \
                        area_ratio < 0.5 and \
                        width_ratio < 0.8 and \
                        height_ratio < 0.2:
                list_matching_chars.append(pos_matching)
                count += 1
                # print('haha' + str(count))
    # print('end')
    if len(list_matching_chars) > 3:
        return list_matching_chars
    return []

def possible_license_plate(contours):
	pos_char = []
	for c in contours:
		contour = Functions.Contour(c)
		if Functions.checkIfChar(contour):
			pos_char.append(contour)
	# print([x.centerX for x in pos_char])
	pos_char = sorted(pos_char, key=lambda x: x.centerX)
	# print([x.centerX for x in pos_char])
	check_list = []
	license_list = []
	for c in pos_char:
		if c in check_list:
			continue
		list_matching_chars = matching_chars(c, pos_char, check_list)
		if list_matching_chars:
			license_list.append(list_matching_chars)
			check_list.extend(list_matching_chars)
	return [get_plate_detail(contours) for contours in license_list]

def get_plate_detail(contours):
	plate = Functions.Plate(contours)
	x1 = np.min([x.boundingRectX for x in plate.contours])
	y1 = np.min([x.boundingRectY for x in plate.contours])
	x2 = np.max([(x.boundingRectX + x.boundingRectWidth) for x in plate.contours])
	y2 = np.max([(x.boundingRectY + x.boundingRectHeight) for x in plate.contours])
	plate.box = (x1, y1, x2, y2)
	plate.width = x2 - x1 
	plate.height = y2 - y1 
	plate.area = plate.width * plate.height
	plate.center_x = (x1 + x2) / 2
	plate.center_y = (y1 + y2) / 2
	return plate

def detect_plate(img):
	# img = cv2.imread(img_path)
	width, height, chanel = img.shape
	contours, threshold = preprocess(img)
	# show_img(threshold)
	possible_plate_list = possible_license_plate(contours)
	return possible_plate_list

if __name__=='__main__':
	img_path = 'test.jpg'
	img = cv2.imread(img_path)
	plate_list = detect_plate(img)

	for plate in plate_list:
		contours = [x.contour for x in plate.contours]
		img = cv2.imread(img_path)
		width, height, chanel = img.shape
		img_contours = np.zeros((width, height, chanel))
		cv2.drawContours(img_contours, contours, -1, (0, 255, 255))
		show_img(img_contours)

		draw_bbox(img, plate.get_extend_box())
		show_img(img)