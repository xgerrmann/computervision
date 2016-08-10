#!/usr/bin/python

import cv2
import dlib
import numpy as np
from paths import trained_model, default_image
from datetime import datetime
from time import time
import math
from utils import pose
import sys
import copy
from homography import hom3

## Sources
# Example:		http://dlib.net/face_landmark_detection.py.html
# Speeding up:	http://www.learnopencv.com/speeding-up-dlib-facial-landmark-detector/
def draw_polyline(img, shape, start, stop, isClosed=False):
# directly from: http://www.learnopencv.com/speeding-up-dlib-facial-landmark-detector/
	points = np.zeros((stop-start,2),dtype=np.int32)
	for i in range(start,stop):
		points[i-start,0] = shape.part(i).x
		points[i-start,1] = shape.part(i).y
	cv2.polylines(img,np.int32([points]),isClosed,[255,0,0])

def showshape(window_face,frame,shape):
# directly from: http://www.learnopencv.com/speeding-up-dlib-facial-landmark-detector/
	draw_polyline(frame, shape, 0, 16);           # Jaw line
	draw_polyline(frame, shape, 17, 21);          # Left eyebrow
	draw_polyline(frame, shape, 22, 26);          # Right eyebrow
	draw_polyline(frame, shape, 27, 31);          # Nose bridge
	draw_polyline(frame, shape, 30, 35, True);    # Lower nose
	draw_polyline(frame, shape, 36, 41, True);    # Left eye
	draw_polyline(frame, shape, 42, 47, True);    # Right Eye
	draw_polyline(frame, shape, 48, 59, True);    # Outer lip
	draw_polyline(frame, shape, 60, 67, True);    # Inner lipt
	cv2.imshow(window_face,frame)
	cv2.waitKey(1)
	return 0

def detect_face(window_face,window_image,predictor,detector,frame):
	# Ask the detector to find the bounding boxes of each face. The 1 in the
	# second argument indicates that we should upsample the image 1 time. This
	# will make everything bigger and allow us to detect more faces.
	# Detect face
	temp = time()
	# subsampling for faster detection
	subsample = 2.0
	frame_subsample = cv2.resize(frame, (0,0), fx=1/subsample, fy=1/subsample)
	dets = detector(frame_subsample, 0)
	print 'Detect: ',time() - temp, 's'
	if len(dets) == 0: # do nothing if no face has been detected
		return 0
	print("Number of faces detected: {}".format(len(dets)))
	# TODO: use only face with highest detection strength: other faces should be ignored
	# TODO: make a model of the location of the face for faster detection
	# Rescale detected recangle in subsampled image
	left	= int(dets[0].left()*subsample)
	top		= int(dets[0].top()*subsample)
	right	= int(dets[0].right()*subsample)
	bottom	= int(dets[0].bottom()*subsample)
	d		= dlib.rectangle(left,top,right,bottom)
	print("Left: {} Top: {} Right: {} Bottom: {}".format(
		d.left(), d.top(), d.right(), d.bottom()))
	# Get the landmarks/parts for the face in box d.
	temp = time()
	shape = predictor(frame, d)
	print 'Predict: ',time() - temp, 's'
	# Draw the face landmarks on the screen.
	temp = time()
	showshape(window_face,frame,shape)
	print 'Draw landmarks: ',time() - temp, 's'

	return shape

def shape2pose(shape_calibrated, shape_current):
# This function determines the pose of the head by using the foudn markers on the face.
# The current markers are compared to the calibrated markers and from this the head pose is determined.
# Currently only the nos bridge length is used, which is only to allow for a proof of concept.
# Definition of head pose: x, y, z, rx (pitch), ry (yaw), rz (roll) -> http://gi4e.unavarra.es/databases/hpdb/
	# Initialization of head pose
	headpose = pose()

	pts		= shape_current.parts()
	#for pt in pts:
	#	print pt
	nose_bridge	= pts[27:31]
	jaw_line	= pts[0:16]
	
	# TODO: no rotation is assumed
	ymin = np.inf
	ymax = -1
	for pt in nose_bridge:
		x = pt.x
		y = pt.y
		if y<ymin:
			ymin = y
		if y>ymax:
			ymax = y
	# TODO: use calibrated shape do determine normal nose bridge length
	nb_length_cal = 35
	nb_length_cur = ymax - ymin

	print 'Bridge length:, ',nb_length_cur
	print 'Normal length:  ',nb_length_cal
	headpose.rx = -math.acos(float(nb_length_cur)/float(nb_length_cal))
	#print	math.acos(float(nb_length_cur)/float(nb_length_cal))

	return headpose

def adjustwindow(windowname,image,headpose):
	rx = headpose.rx
	ry = headpose.ry
	rz = headpose.rz
	print 'rx: ',rx
	print 'ry: ',ry
	print 'rz: ',rz
	image_adj = hom3(image,rx,ry,rz)
	cv2.imshow('Image adjusted',image_adj)
	cv2.waitKey(1)
	return 0

def main():
	detector	= dlib.get_frontal_face_detector()
	predictor	= dlib.shape_predictor(trained_model)
	image		= cv2.imread(default_image)
	window_image = "Image"
	cv2.imshow(window_image,image)
	cv2.waitKey(1)
	window_face = 'Face'
	cv2.namedWindow(window_face)
	vc = cv2.VideoCapture(0)
	#win_dlib = dlib.image_window()
	if vc.isOpened(): # try to get the first frame
		rval, frame = vc.read()
	else:
		sys.exit('No frame captured')

	while True:
		#cv2.imshow("preview", frame)
		rval, frame = vc.read()
		key = cv2.waitKey(1)
		if key == 27: # exit on ESC
			break
		shape = detect_face(window_face,window_image,predictor,detector,frame)
		if shape == 0: # In the case of no detection, continue to the next frame
			cv2.imshow(window_face,frame)
			cv2.waitKey(1)
			continue
		else:
			# TODO: calibrated shape is now same as current, calibrated shape must be indicated by used.
			shape_calibrated= shape
			shape_current	= shape
			headpose		= shape2pose(shape_calibrated, shape_current)
			adjustwindow(window_image, image, headpose)

	vc.release()

	return 0

if __name__ == '__main__':
	main()
