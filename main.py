#!/usr/bin/python

import cv2
import dlib
import numpy as np
from paths import trained_model

def show_landmarks(win,predictor,detector,frame):
	# Ask the detector to find the bounding boxes of each face. The 1 in the
	# second argument indicates that we should upsample the image 1 time. This
	# will make everything bigger and allow us to detect more faces.
	dets = detector(frame, 1)
	print("Number of faces detected: {}".format(len(dets)))
	# TODO: use only face with highest detection strength: other faces should be ignored
	print dets
	win.clear_overlay()
	if len(dets)>0:
		d	= dets[0]
		k	= 0
		#print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
		#	k, d.left(), d.top(), d.right(), d.bottom()))
		# Get the landmarks/parts for the face in box d.
		shape = predictor(frame, d)
		#print("Part 0: {}, Part 1: {} ...".format(shape.part(0),shape.part(1)))
		# Draw the face landmarks on the screen.
		win.add_overlay(shape)
	win.set_image(frame)
	#print np.array(dets)

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
		show_landmarks(win,predictor,detector,frame)
		#dlib.hit_enter_to_continue()
	vc.release()
	#cv2.destroyWindow("preview")

	return 0

if __name__ == '__main__':
	main()
