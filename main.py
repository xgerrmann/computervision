#!/usr/bin/python

import cv2
import dlib
import numpy as np
from paths import trained_model
from datetime import datetime
from time import time

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

	return 0

def main():
	detector	= dlib.get_frontal_face_detector()
	print trained_model
	predictor	= dlib.shape_predictor(trained_model)
	#cv2.namedWindow("preview")
	vc = cv2.VideoCapture(0)
	win = dlib.image_window()
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
		detect(win,predictor,detector,frame)
	vc.release()
	#cv2.destroyWindow("preview")

	return 0

if __name__ == '__main__':
	main()
