#!/usr/bin/env python3
# Laboratório de Visão Computacional - EP 01
# Nome: Lucas Martinuzzo Batista <br>

# Imports
import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
from scipy import ndimage
import sys
import os

class dummyKeypoint():
    def __init__(self,width,height):
        self.pt = [width/2,height/2]
        
        
def markersDetector(image):
    markers_params = cv2.SimpleBlobDetector_Params()
    #Extracting only black blobs
    markers_params.filterByColor = True
    markers_params.blobColor = 0
    # Extracting blobs with area between minArea (inclusive) and maxArea (exclusive).
    markers_params.filterByArea = True
    markers_params.minArea = 300
    markers_params.maxArea = 500
    
    markers_detector = cv2.SimpleBlobDetector_create(markers_params)
    marker_keypoints = markers_detector.detect(image)
    height, width = image.shape
    top_left_keypoint = dummyKeypoint(width,height)
    top_right_keypoint = dummyKeypoint(width,height)
    bottom_left_keypoint = dummyKeypoint(width,height)
    bottom_right_keypoint = dummyKeypoint(width,height)
    filtered_marker_keypoints = []
    for keypoint in marker_keypoints:
        if keypoint.pt[0] < top_left_keypoint.pt[0] and keypoint.pt[1] < top_left_keypoint.pt[1]:
                top_left_keypoint = keypoint
        if keypoint.pt[0] < bottom_left_keypoint.pt[0] and keypoint.pt[1] > bottom_left_keypoint.pt[1]:
                bottom_left_keypoint = keypoint
        if keypoint.pt[0] > top_right_keypoint.pt[0] and keypoint.pt[1] < top_right_keypoint.pt[1]:
                top_right_keypoint = keypoint
        if keypoint.pt[0] > bottom_right_keypoint.pt[0] and keypoint.pt[1] > bottom_right_keypoint.pt[1]:
                bottom_right_keypoint = keypoint
        #print_keypoint(keypoint)
    filtered_marker_keypoints = [top_left_keypoint,top_right_keypoint,
                                 bottom_left_keypoint,bottom_right_keypoint]
    return filtered_marker_keypoints

def codeDetector(image):
    # Initialize parameter settiing using cv2.SimpleBlobDetector
    code_params = cv2.SimpleBlobDetector_Params()
    #Extracting only white blobs
    code_params.filterByColor = True
    code_params.blobColor = 255
    # Extracted blobs have an area between minArea (inclusive) and maxArea (exclusive).
    code_params.filterByArea = True
    code_params.minArea = 250
    code_params.maxArea = 350
    code_detector = cv2.SimpleBlobDetector_create(code_params)
    # Detect blobs
    code_keypoints = code_detector.detect(image)
    return code_keypoints

def nuspDetector(image):
    #Morphologic operations needed to generate the correct blobs
    height, width = image.shape
    kernel = np.ones((2,2), np.uint8)
    erode = cv2.erode(image, kernel, iterations = 1)
    kernel = np.ones((15,15), np.uint8)
    closing = cv2.morphologyEx(erode, cv2.MORPH_CLOSE, kernel)
    kernel = np.ones((3,3), np.uint8)
    dilate = cv2.dilate(closing, kernel, iterations = 5)
    erode = cv2.erode(dilate, kernel, iterations = 5)
    #Extraction
    nusp_params = cv2.SimpleBlobDetector_Params()
    #Extracting only black blobs
    nusp_params.filterByColor = True
    nusp_params.blobColor = 0
    # Extracted blobs have an area between minArea (inclusive) and maxArea (exclusive).
    nusp_params.filterByArea = True
    nusp_params.minArea = 700
    nusp_params.maxArea = 1000
    nusp_detector = cv2.SimpleBlobDetector_create(nusp_params)
    nusp_keypoints = nusp_detector.detect(erode)
    #Filter only the ones on the top left corner
    nusp_keypoints = [keypoint for keypoint in nusp_keypoints if keypoint.pt[0] < width/2 and keypoint.pt[1] < height/2]
    return nusp_keypoints
    
# For debugging purposes
def printImageWithKeyPoints(image,keypoints):
    # Draw blobs on our image as red circles
    blank = np.zeros((1,1)) 
    blobs = cv2.drawKeypoints(image, keypoints, blank, (255,0,0),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    number_of_blobs = len(keypoints)
    text = "No. of Blobs: " + str(len(keypoints))
    # Show blobs
    plt.title(text)
    figure = plt.gcf() # get current figure
    figure.set_size_inches(16, 23)
    plt.imshow(blobs)
    return

# For debugging purposes
def printKeypoint(keypoint):
    print(keypoint.pt[0], keypoint.pt[1],keypoint.size)
    return

# The code will be read by seing the pixel color of each box measuring the distance between its borders.
def readCode(image,marker_keypoints):
    top_left_keypoint = marker_keypoints[0]
    top_right_keypoint = marker_keypoints[1]
    W = top_right_keypoint.pt[0] - top_left_keypoint.pt[0]
    top_left_code = np.array([top_left_keypoint.pt[0] + W*0.259, top_left_keypoint.pt[1] + W*(-0.0435)])
    bot_left_code = np.array([top_left_keypoint.pt[0] + W*0.259, top_left_keypoint.pt[1] + W*(-0.0208)])
    average_x_diameter = W*(0.474 - 0.259)/12.0
    average_x_radius = average_x_diameter/2
    average_y_diameter = W*(0.0435 - 0.0258)
    average_y_radius = average_y_diameter/2
    upper_binary = ['1','1','1','1','1','1','1','1','1','1','1','1']
    lower_binary = ['1','1','1','1','1','1','1','1','1','1','1','1']
    for k in range(0,12):
        y_top = int(top_left_code[1]  + average_y_radius)
        y_bot = int(bot_left_code[1]  + average_y_radius)
        x = int(top_left_code[0] + average_x_radius + k*average_x_diameter)
        #print(x,y_top,y_bot)
        top_pixels = image[y_top,(x-2):(x+3)]
        #print(top_pixels)
        if np.where(top_pixels == 255)[0].size >= 3:
            upper_binary[k] = '0'
        bot_pixels = image[y_bot,(x-2):(x+3)]
        #print(bot_pixels)
        if np.where(bot_pixels == 255)[0].size >= 3:
            lower_binary[k] = '0'
    return upper_binary, lower_binary


def readNUSP(marker_keypoints,nusp_keypoints):
    top_left_keypoint = marker_keypoints[0]
    top_right_keypoint = marker_keypoints[1]
    W = top_right_keypoint.pt[0] - top_left_keypoint.pt[0]
    top_left_nusp = np.array([top_left_keypoint.pt[0] + W*0.0456, top_left_keypoint.pt[1] + W*(0.111)])
    bottom_right_nusp = np.array([top_left_keypoint.pt[0] + W*0.2484, top_left_keypoint.pt[1] + W*(0.4049)])
    average_y_space = (bottom_right_nusp[1] - top_left_nusp[1])/10
    #half_y_space = average_y_space/2
    average_x_space = (bottom_right_nusp[0] - top_left_nusp[0])/8
    #half_x_space = average_x_space/2
    nusp = [0,0,0,0, 0,0,0,0]
    for keypoint in nusp_keypoints:
        #print_keypoint(keypoint)
        if keypoint.pt[0] > top_left_nusp[0] and keypoint.pt[0] < bottom_right_nusp[0]:
            if keypoint.pt[1] > top_left_nusp[1] and keypoint.pt[1] < bottom_right_nusp[1]:
                position = int((keypoint.pt[0] - top_left_nusp[0])/average_x_space)
                number = int((keypoint.pt[1] - top_left_nusp[1])/average_y_space)
                #print(position,number)
                nusp[position] = number
    #print(nusp)
    return "".join([str(n) for n in nusp])

def binaryListToDecimal(binary_list):
    return int("".join(binary_list),2)

def checkVerificationNumber(test_number,page_number):
    return 60 - ((test_number - 1)*4 + (page_number -1))%60

def main():
    assert len(sys.argv) == 2, """You should pass the image path when calling the program.
    For example: ./ep01.py ./scans/mac2166-t8.PDF-page-001-000.pbm"""
    image_path = sys.argv[1]
    if not os.path.isfile(image_path):
        print("File not found.")
        return
    original_image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    # Resizing to as close as 1324 width x 1872 height as possible
    larger_side = max(original_image.shape)
    scale_percent = 1872.0/larger_side
    new_dim = (int(original_image.shape[1]*scale_percent),
               int(original_image.shape[0]*scale_percent))
    original_image = cv2.resize(original_image, new_dim, interpolation = cv2.INTER_AREA)
    min_color = np.min(original_image)
    max_color = np.max(original_image)
    #Making sure black is 0 and white is 255
    original_image = ((original_image - min_color)/(max_color - min_color)*255).astype(np.uint8)
    # Binarizing
    _,image = cv2.threshold(original_image,200,255,cv2.THRESH_BINARY)
    height, width = image.shape
    # Getting marker blobs
    marker_keypoints = markersDetector(image)
    # Checking if the test is crooked and  and correcting the angle by calculating
    # the arctan between the points on the left
    top_left_keypoint = marker_keypoints[0]
    bottom_left_keypoint = marker_keypoints[2]
    diff_y = np.abs(top_left_keypoint.pt[1] - bottom_left_keypoint.pt[1])
    diff_x = top_left_keypoint.pt[0] - bottom_left_keypoint.pt[0]
    #print(diff_x, diff_y)
    degree = math.degrees(math.atan2(diff_x,diff_y))
    #0.01 Value found by empiric observation
    if np.abs(degree) >= 0.01:
        image = ndimage.rotate(image,degree)
        height, width = image.shape
        marker_keypoints = markersDetector(image)
    #Getting code blobs
    code_keypoints = codeDetector(image)
    #Checking if test have vertical orientation
    is_vertical = height > width
    #Finding out if k*90° rotation is needed
    mean_code_x = np.mean([keypoint.pt[0] for keypoint in code_keypoints])
    mean_code_y = np.mean([keypoint.pt[1] for keypoint in code_keypoints])
    rotate_degree_clockwise = 0
    if is_vertical:
        if mean_code_y < height/2:
            rotate_degree_clockwise = 0
        else:
            rotate_degree_clockwise = 180
    else:
        if mean_code_x < width/2:
            rotate_degree_clockwise = 270
        else:
            rotate_degree_clockwise = 90
    #Rotating if needed
    if rotate_degree_clockwise > 0:
        image = ndimage.rotate(image,rotate_degree_clockwise)
        height, width = image.shape
        marker_keypoints = markersDetector(image)
    code_keypoints = codeDetector(image)
    #If by chance any other blob is detected, remove it
    filtered_code_keypoints = []
    for keypoint in code_keypoints:
        if keypoint.pt[1] < top_left_keypoint.pt[1]:
            filtered_code_keypoints.append(keypoint)
            #printKeypoint(keypoint)
    code_keypoints = filtered_code_keypoints
    upper_binary, lower_binary = readCode(image,marker_keypoints)
    test_number = binaryListToDecimal(upper_binary)
    page_number = binaryListToDecimal(lower_binary[:6])
    verification_number = binaryListToDecimal(lower_binary[6:])
    calculated_verification_number = checkVerificationNumber(test_number,page_number)
    if calculated_verification_number != verification_number:
        print("Calculated verification number {} doesn't match the one read {}".format(
            calculated_verification_number,verification_number))
        print('test number:',test_number)
        print('page:',page_number)
    else:
        print("Sucessful detection:")
        print('test number:',test_number)
        print('page: {}, code: {}'.format(page_number,verification_number))
    if page_number == 1:
        nusp_keypoints = nuspDetector(image)
        nusp = readNUSP(marker_keypoints,nusp_keypoints)
        print('nusp:',nusp)
    return 0

if __name__ == "__main__":
    main()