# -*- coding: utf-8 -*-
#!/usr/bin/env python3

# Laboratório de Visão Computacional - EP 02
# Nome: Lucas Martinuzzo Batista

# Imports
import cv2 as cv
import numpy as np
import math
import sys
import os
from scipy import ndimage

VIDEOS = {'0': 'motociata.webm',
          '1': 'road_traffic_1.mp4',
          '2': 'road_traffic_2.mp4',
          '3a': 'road_traffic_3.mp4',
          '3b': 'road_traffic_3.mp4',
          '4': 'road_traffic_4.mp4',
          '5a': 'road_traffic_5.mp4',
          '5b': 'road_traffic_5.mp4',
          }

ARG_ERROR = """
    Since the program is made to track an especific region, of the image, 
    you must especify the index of the given videos when calling the program:
    - 0 : motociata
    - 1 : road_traffic_1
    - 2 : road_traffic_2
    - 3a : road_traffic_3 left side
    - 3b : road_traffic_3 right side
    - 4 : road_traffic_4
    - 5a : road_traffic_5 left side
    - 5b : road_traffic_5 right side
    For example: python ep02.py 1 or python ep02.py 3a
    The videos must be located in the folder ./videos with the format .mp4.
    """

N_FRAMES_TO_APPEAR = 15
N_FRAMES_TO_DISAPPEAR = 20
MAX_DISTANCE = 50


def calculateCenter(bounding_box):
    x, y, w, h = bounding_box 
    cx = x + w//2
    cy = y + h//2
    center = (cx,cy)
    return center



class Detection():
    def __init__(self,bounding_box):
        """
        The Detection Class saves a bounding box detection and all the information
        that helps tracking it.
        Parameters
        ----------
        bounding_box : (int,int,int,int)
            Bounding box of the object (x,y,w,h)

        Returns
        -------
        None.

        """
        self.bounding_box = bounding_box
        #Center coordinates of the object
        self.center = calculateCenter(bounding_box)
        # Number of frames that the object appeared
        self.n_frames_appeared = 1
        # Number of consecutive frames that the object didn't appeared
        self.n_frames_not_appeared = 0
        # Informs if the object appeared on the ROI on the current frame
        self.appeared = True
        # Bounding box of the object
    
    def _updateCoordinates(self,bounding_box):
        self.bounding_box = bounding_box
        self.center = calculateCenter(bounding_box)
        return        
    
    def calculateDistance(self,cx,cy):
        distance = math.hypot(self.center[0] - cx, self.center[1] - cy)
        return distance
    
    def updateDetection(self,bounding_box):
        self._updateCoordinates(bounding_box)
        self.n_frames_appeared+=1
        self.appeared = True
        self.n_frames_not_appeared = 0
        return
        
    def getCenter(self):
        return self.center
    
    def getBoundingBox(self):
        return self.bounding_box
    
    def getNFramesAppeared(self):
        return self.n_frames_appeared

    # Check if the detection have appeared in the current frame
    def haveAppeared(self):
        return self.appeared
    
    #Update the counter if the detection haven't appeared and 
    # resets the checking variable.
    def updateIfNotAppeared(self):
        if self.appeared is False:
            self.n_frames_not_appeared+=1
        self.appeared = False
        return
    
    def getNFramesNotAppeared(self):
        return self.n_frames_not_appeared
    
        
class DetectionTracker():
    def __init__(self,n_frames=N_FRAMES_TO_APPEAR,distance=MAX_DISTANCE):
        """
        The DetectionTracker class counts how many valid detections were made in the video.
        
        Parameters
        ----------
        n_frames : int, optional
            Sets the minimum number of frames an object have to be located in
            order to count. The default is N_FRAMES_TO_APPEAR.

        distance : int, optional
            Sets the max distance that two objects have to have from one to another 
            to be considered the same object. The default is MAX_DISTANCE.
        Returns
        -------
        None.
        """
        self.n_frames_to_disappear = n_frames
        self.distance = distance
        # All the valid objects found (exists for n_frames or more) will be
        # stored here
        self.valid_detections = {}
        # All the objects found
        self.temporary_detections = {}
        # Hash between temp_id and valid_id
        self.hash_ids = {}
        # Counts the valid objects detected
        self.valid_count = 0
        # Counts the total objects detected
        self.all_count = 0

    def getValidCount(self):
        return self.valid_count
    
    # List only the valid objects that appeared in the current frame
    def _getVisibleDetections(self):
        appeared_detections = {}
        for valid_id,detection in self.valid_detections.items():
            if detection.haveAppeared():
                appeared_detections[valid_id] = detection
        return appeared_detections

    # Check all the objects found and remove the ones that haven't appeared for
    # n_frames
    def _removeUnseenDetections(self):
        temp_ids = list(self.temporary_detections.keys())
        for temp_id in temp_ids:
            detection = self.temporary_detections[temp_id]
            detection.updateIfNotAppeared()
            #If the objected havent appeared for n_frames
            if detection.getNFramesNotAppeared() == self.n_frames_to_disappear:
                self.temporary_detections.pop(temp_id)
                if temp_id in self.hash_ids:
                    valid_id = self.hash_ids[temp_id]
                    self.valid_detections.pop(valid_id)
                    
            

    def update(self,objects_detected):
        for bounding_box in objects_detected:
            cx, cy = calculateCenter(bounding_box)
            object_already_detected = False
            #For each objected already encountered
            for temp_id, temp_obj in self.temporary_detections.items():
                distance = temp_obj.calculateDistance(cx,cy)
                #Check if the bounding_box checks with already found objects
                if distance < self.distance:
                    object_already_detected = True
                    temp_obj.updateDetection(bounding_box)
                    n_frames = temp_obj.getNFramesAppeared()
                    #If found in n_frames, the object is now valid to show up
                    if n_frames == self.n_frames_to_disappear:
                        self.valid_count+=1
                        self.valid_detections[self.valid_count] = temp_obj
                        self.hash_ids[temp_id] = self.valid_count
                    break
            if object_already_detected is False:
                new_detection = Detection(bounding_box)
                self.all_count+=1
                self.temporary_detections[self.all_count] = new_detection
        appeared_detections = self._getVisibleDetections()
        self._removeUnseenDetections()
        return appeared_detections
                            

def runHistogramEqualization(image):
    img_yuv = cv.cvtColor(image, cv.COLOR_BGR2YUV)
    clahe = cv.createCLAHE(clipLimit=6.0, tileGridSize=(2,2))
    img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
    #img_yuv[:,:,0] = cv.equalizeHist(img_yuv[:,:,0])
    equalized_img = cv.cvtColor(img_yuv, cv.COLOR_YUV2BGR)
    return equalized_img


def detectorROI(video_id):
    if video_id == '0':
        detector = cv.createBackgroundSubtractorMOG2(history = 50,
                                                  varThreshold=40)
    elif video_id == '1':
        detector = cv.createBackgroundSubtractorMOG2(history = 100,
                                                  varThreshold=40)
    elif video_id == '2':
         detector = cv.createBackgroundSubtractorMOG2(history = 150,
                                                  varThreshold=50)
    elif video_id == '3a':
         detector = cv.createBackgroundSubtractorMOG2(history = 150,
                                                  varThreshold=30)
    elif video_id == '3b':
         detector = cv.createBackgroundSubtractorMOG2(history = 150,
                                                  varThreshold=40)
    elif video_id == '4':
         detector = cv.createBackgroundSubtractorMOG2(history = 150,
                                                  varThreshold=50)
    elif video_id == '5a':
         detector = cv.createBackgroundSubtractorMOG2(history = 150,
                                                  varThreshold=50)
    elif video_id == '5b':
         detector = cv.createBackgroundSubtractorMOG2(history = 150,
                                                  varThreshold=50)
    return detector

def thresholdArea(video_id):
    if video_id == '0':
        return 200
    if video_id == '1':
        return 100
    if video_id == '2':
        return 500
    if video_id == '3a':
        return 500
    if video_id == '3b':
        return 500
    if video_id == '4':
        return 300
    if video_id == '5a':
        return 300
    if video_id == '5b':
        return 300
    return 100


def videoROI(frame,video_id):
    if video_id == '0':
        return frame[700: 1000,550: 1165]
    if video_id == '1':
        return frame[340: 720,500: 800]
    if video_id == '2':
        return frame[340: 720, 0: 800]
    if video_id == '2':
        return frame[340: 720, 0: 800]
    if video_id == '3a':
        return frame[350: 600, 100: 600]
    if video_id == '3b':
        return frame[350: 600, 700: 1200]
    if video_id == '4':
        return frame[350: 700, 230: 900]
    if video_id == '5a':
        return frame[350: 700, 100: 600]
    if video_id == '5b':
        return frame[350: 700, 650: 1200]
    return frame


def createMask(roi,detector,video_id):
    if video_id == '0':
        mask = roi.copy()
        mask = cv.GaussianBlur(mask,(3,3),0)
        mask = detector.apply(mask)
        _, mask = cv.threshold(mask,254, 255, cv.THRESH_BINARY)
        closing_kernel = np.ones((11,11),np.uint8)
        erosion_kernel = np.ones((2,2),np.uint8)
        dilate_kernel =  np.ones((3,3),np.uint8)
        opening_kernel = np.ones((3,3),np.uint8)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, closing_kernel)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, opening_kernel)
    elif video_id == '1':
        mask = detector.apply(roi)
        closing_kernel = np.ones((7,7),np.uint8)
        erosion_kernel = np.ones((3,3),np.uint8)
        mask = cv.GaussianBlur(mask,(3,9),0)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, closing_kernel)
        mask = cv.erode(mask,erosion_kernel,iterations = 1)
        _, mask = cv.threshold(mask,200, 255, cv.THRESH_BINARY)
    elif video_id == '2':
        mask = roi.copy()
        mask = cv.GaussianBlur(mask,(3,3),0)
        mask = detector.apply(mask)
        _, mask = cv.threshold(mask,200, 255, cv.THRESH_BINARY)
        closing_kernel = np.ones((20,20),np.uint8)
        erosion_kernel = np.ones((3,3),np.uint8)
        opening_kernel = np.ones((2,2),np.uint8)
        dilate_kernel =  np.ones((3,3),np.uint8)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, opening_kernel)
        mask = cv.dilate(mask,dilate_kernel,iterations = 1)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, closing_kernel)
        #mask = cv.erode(mask,erosion_kernel,iterations = 1)
    elif video_id == '3a':
        mask = roi.copy()
        mask = cv.GaussianBlur(mask,(3,3),0)
        mask = detector.apply(mask)
        _, mask = cv.threshold(mask,200, 255, cv.THRESH_BINARY)
        closing_kernel = np.ones((7,7),np.uint8)
        erosion_kernel = np.ones((3,3),np.uint8)
        opening_kernel = np.ones((2,2),np.uint8)
        dilate_kernel =  np.ones((3,3),np.uint8)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, closing_kernel)
    elif video_id == '3b':
        mask = roi.copy()
        mask = cv.GaussianBlur(mask,(3,3),0)
        mask = detector.apply(mask)
        _, mask = cv.threshold(mask,200, 255, cv.THRESH_BINARY)
        closing_kernel = np.ones((7,7),np.uint8)
        erosion_kernel = np.ones((3,3),np.uint8)
        opening_kernel = np.ones((2,2),np.uint8)
        dilate_kernel =  np.ones((3,3),np.uint8)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, closing_kernel)
    elif video_id == '4':
        mask = roi.copy()
        mask = cv.GaussianBlur(mask,(3,3),0)
        mask = detector.apply(mask)
        _, mask = cv.threshold(mask,200, 255, cv.THRESH_BINARY)
        closing_kernel = np.ones((20,20),np.uint8)
        erosion_kernel = np.ones((3,3),np.uint8)
        opening_kernel = np.ones((2,2),np.uint8)
        dilate_kernel =  np.ones((3,3),np.uint8)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, closing_kernel)
        mask = cv.erode(mask,erosion_kernel,iterations = 1)
    elif video_id == '5a':
        mask = roi.copy()
        mask = cv.GaussianBlur(mask,(3,3),0)
        mask = detector.apply(mask)
        _, mask = cv.threshold(mask,200, 255, cv.THRESH_BINARY)
        closing_kernel = np.ones((20,20),np.uint8)
        erosion_kernel = np.ones((3,3),np.uint8)
        opening_kernel = np.ones((2,2),np.uint8)
        dilate_kernel =  np.ones((3,3),np.uint8)
    elif video_id == '5b':
        mask = roi.copy()
        mask = cv.GaussianBlur(mask,(3,3),0)
        mask = detector.apply(mask)
        _, mask = cv.threshold(mask,200, 255, cv.THRESH_BINARY)
        closing_kernel = np.ones((20,20),np.uint8)
        erosion_kernel = np.ones((3,3),np.uint8)
        opening_kernel = np.ones((2,2),np.uint8)
        dilate_kernel =  np.ones((3,3),np.uint8)
    return mask



def main():
    assert len(sys.argv) == 2, ARG_ERROR
    if sys.argv[1] in VIDEOS.keys():
        video_id = sys.argv[1]
        video = VIDEOS[video_id]
    else:
        print(ARG_ERROR)
        return -1
    
    video_path = os.path.join("./videos",video)
    if not os.path.isfile(video_path):
        print("File not found. Make sure the videos are in the folder ./videos",
              "with the correct name and the format .mp4.")
        return -1
    
    capture = cv.VideoCapture(video_path)
    detector = detectorROI(video_id)
    tracker = DetectionTracker()
    while True:
        retval,frame = capture.read()
        if not retval:
            break
        roi = videoROI(frame,video_id)
        # Object detection
        mask = createMask(roi,detector,video_id)
        contours, _ = cv.findContours(mask,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
        detections = []
        for contour in contours:
            area = cv.contourArea(contour)
            if area > thresholdArea(video_id):
                x,y,w,h = cv.boundingRect(contour)
                detections.append([x,y,w,h])
        # Tracking
        detections = tracker.update(detections)
        for detection_id, detection in detections.items():
            x,y,w,h = detection.getBoundingBox()
            cv.putText(roi, str(detection_id), (x, y - 15), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            cv.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)
        
        cv.imshow("roi", roi)
        cv.imshow("Frame", frame)
        cv.imshow("Mask", mask)

        key = cv.waitKey(10)
        if key == 27:
            break

    capture.release()
    cv.destroyAllWindows()
        
    print("Total of vehicles:",tracker.getValidCount())
    return 0

if __name__ == "__main__":
    main()