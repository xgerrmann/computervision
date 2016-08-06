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

## Sources
# Example:		http://dlib.net/face_landmark_detection.py.html
# Speeding up:	http://www.learnopencv.com/speeding-up-dlib-facial-landmark-detector/

def detect(win,predictor,detector,frame):
	# Ask the detector to find the bounding boxes of each face. The 1 in the
	# second argument indicates that we should upsample the image 1 time. This
	# will make everything bigger and allow us to detect more faces.
	# Draw new frame
	temp = time()
	win.set_image(frame)
	print 'Plot frame: ',time() - temp, 's'
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
	# Clear overlay
	temp = time()
	win.clear_overlay()
	print 'Clear overlay: ',time() - temp, 's'
	# Draw the face landmarks on the screen.
	temp = time()
	win.add_overlay(shape)
	print 'Add overlay: ',time() - temp, 's'

	return shape

# TODO: draw in openCV and not wit dlib (shape.parts())
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
	nb_length_cal = 45
	nb_length_cur = ymax - ymin

	headpose.rx = math.cos(float(nb_length_cur)/float(nb_length_cal))
	print nb_length_cur
	print nb_length_cal
	print	math.acos(float(nb_length_cur)/float(nb_length_cal))

	return headpose

def adjustwindow(windowname,image,headpose):
	a = 1
	b = 0
	c = 0
	d = 0
	e = 1
	f = 0
	g = 0
	h = 0
	i = 1
	homography	= np.array([[a,b,c],[d,e,f],[g,h,i]])
	hom_inv		= np.linalg.inv(homography)
	print homography
	print hom_inv
	print image.shape
	(height, width, channels) = image.shape

	w = 1
	X = np.zeros((height,width))
	Y = np.zeros((height,width))
	W = np.zeros((height,width))
	for x in range(width):
		print x
		for y in range(height):
			temp = np.matrix([[x],[y],[w]])
			#print temp
			res = np.dot(hom_inv,temp)
			#print res
			X[y,x] = res[0]
			Y[y,x] = res[1]
			W[y,x] = res[2]
	image_new = np.zeros((height,width,channels),dtype=np.uint8)
	for x in range(width):
		for y in range(height):
			# TODO: perform interpolation instead of rounding
			x_tmp = int(X[y,x])
			y_tmp = int(Y[y,x])
			#print 'x: ',x,'-> ',x_tmp
			#print 'y: ',y,'-> ',y_tmp
			if x_tmp > width or y_tmp >height or x_tmp<0 or y_tmp < 0:
				image_new[y,x,:] = 0
			else:
				#print 'Before'
				#print image[y,x,:]
				#print image_new[y,x,:]
				image_new[y,x,:] = copy.deepcopy(image[y_tmp,x_tmp,:])
				#print 'After'
				#print image[y,x,:]
				#print image_new[y,x,:]

				#image_new = image
	cv2.imshow('test',image_new)
	cv2.waitKey(1)
	return 0

def main():
	detector	= dlib.get_frontal_face_detector()
	print trained_model
	predictor	= dlib.shape_predictor(trained_model)
	image		= cv2.imread(default_image)
	windowname = "image"
	cv2.namedWindow(windowname)
	cv2.namedWindow('test')
	cv2.imshow(windowname,image)
	cv2.waitKey(1)
	vc = cv2.VideoCapture(0)
	win_dlib = dlib.image_window()
	if vc.isOpened(): # try to get the first frame
		rval, frame = vc.read()
	else:
		rval = False

	while rval:
		#cv2.imshow("preview", frame)
		rval, frame = vc.read()
		key = cv2.waitKey(1)
		if key == 27: # exit on ESC
			break
		shape = detect(win_dlib,predictor,detector,frame)
		if shape == 0:
			continue
		else:
			# TODO: calibrated shape is now same as current, calibrated shape must be indicated by used.
			shape_calibrated= shape
			shape_current	= shape
			headpose		= shape2pose(shape_calibrated, shape_current)
			print headpose
			adjustwindow(windowname, image, headpose)

	vc.release()
	#cv2.destroyWindow("preview")

	return 0

if __name__ == '__main__':
	main()
