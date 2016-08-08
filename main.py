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

def calcrotationmatrix(rx, ry, rz):
	# source: http://nghiaho.com/?page_id=846
	# source: https://en.wikipedia.org/wiki/3D_projection (uses negative angles?)
	#rx = -rx
	ry = -ry
	rz = -rz
	X	= np.matrix([[1, 0, 0],
					[0,	math.cos(rx), -math.sin(rx)],
					[0, math.sin(rx), math.cos(rx)]])
	Y	= np.matrix([[math.cos(ry), 0, math.sin(ry)],
					[0, 1, 0],
					[-math.sin(ry), 0, math.cos(ry)]])
	Z	= np.matrix([[math.cos(rz), -math.sin(rz), 0],
					[math.sin(rz), math.cos(rz), 0],
					[0, 0, 1]])
#	# source (1): http://web.cs.iastate.edu/~cs577/handouts/homogeneous-transform.pdf
#	# source (2) http://www.cs.virginia.edu/~gfx/Courses/2012/IntroGraphics/lectures/10-Transformations.pdf
#	X	= np.matrix([[1, 0,				0,				0],
#					[0,	 math.cos(rx),	-math.sin(rx),	0],
#					[0,  math.sin(rx),	math.cos(rx),	0],
#					[0,  0,				0,				1]])
#	Y	= np.matrix([[math.cos(ry),	0,	math.sin(ry),	0],
#					[0,				1,	0,				0],
#					[-math.sin(ry),	0,	math.cos(ry),	0],
#					[0,				0,	0,				1]])
#	Z	= np.matrix([[math.cos(rz),	-math.sin(rz),	0,	0],
#					[math.sin(rz),	math.cos(rz),	0,	0],
#					[0,				0,				1,	0],
#					[0,				0,				0,	1]])
	R	= Z*Y*X
	# rotation matrix applies only on X and Y, there is no Z. Therefore
	# remove last row and column
	#R	= R[0:2,0:2]
	return R

def getoutputimagesize(Tpers,homography,height,width):
	# Corner points : [x,y,w]
#	p0 = np.matrix([[0],[0],[1]])
#	p1 = np.matrix([[width],[0],[1]])
#	p2 = np.matrix([[0],[height],[1]])
#	p3 = np.matrix([[width],[height],[1]])
	# with z coordinate = 0
	# w = 1
	p0 = np.matrix([[0],[0],[0],[1]])
	p1 = np.matrix([[width],[0],[0],[1]])
	p2 = np.matrix([[0],[height],[0],[1]])
	p3 = np.matrix([[width],[height],[0],[1]])
	cornerpoints = [p0,p1,p2,p3]
	xmin = np.inf
	xmax = -np.inf
	ymin = np.inf
	ymax = -np.inf
	for p in cornerpoints:
		res = Tpers*homography*p
		print p
		print res
		x_tmp = res[0]
		y_tmp = res[1]
		if x_tmp<xmin:
			xmin=x_tmp
		if y_tmp<ymin:
			ymin=y_tmp
		if x_tmp>xmax:
			xmax=x_tmp
		if y_tmp>ymax:
			ymax=y_tmp
	height_out	= int(np.ceil(ymax-ymin))
	width_out	= int(np.ceil(xmax-xmin))
	print 'Size image_out: ',height_out,', ',width_out
	return (height_out, width_out)

def adjustwindow(windowname,image,headpose):
	rx	= 0.0*math.pi
	ry	= 0.0*math.pi
	rz	= 0.0*math.pi
	R	= calcrotationmatrix(rx, ry, rz)
	
#	tx	= 0 # translation x
#	ty	= 0 # translation y
#	tz	= 0 # translation z

	f = 0.4 # TODO: must be detected accurately for correct results

	# e: viewers position relative to the camera
	ex = 0
	ey = 0
	ez = -f
	e = np.matrix([[-f],[0],[0]]) # viewer is direcly behind projection plane
	# T: transformation matrix for projection of 3D -> 2D
	T = np.zeros((4,4))
	T[0,0] = 1
	T[1,1] = 1
	T[2,2] = 1
	T[0,2] = -ex/ez
	T[1,2] = -ey/ez
	T[3,2] = 1/ez

	print 'Rotation matrix: \n',R
	print 'Transformation matrix: \n',T
	
#	hom_inv		= np.linalg.inv(homography)
#	print 'Inverse homography:\n',hom_inv
#	print image.shape
	(height, width, channels) = image.shape

	# determine size of resulting image:
	#(height_out, width_out) = getoutputimagesize(T,R,height,width)
	height_out	= height
	width_out	= width
	pitch = 0.2625*(10**-3) # [m] pixel pitch (pixel size) assume square pixels, which is generally true
	print 'Pitch: %.6f'%pitch
	w = 1
	X = np.zeros((height_out,width_out))
	Y = np.zeros((height_out,width_out))
	Y = np.zeros((height_out,width_out))
	W = np.zeros((height_out,width_out))
	for xp in range(width_out):
		#print x
		for yp in range(height_out):
			# a is the position of the point to be projected
			# x and y depend on pixel size (screen dependent)
			x = xp*pitch
			y = yp*pitch
			a = np.matrix([[x],[y],[0]]) # virtual image is always at z=0
			print 'a: ',a
			# c is the 3D position of the camera
			c = np.matrix([[0],[0],[0]])
			print 'c: ',c
			# d is the position of point A with respect to a coordinate system
			dtmp = R*(a-c)
			# convert d to homogeneous coordinates
			print dtmp
			d = np.matrix(np.zeros((4,1)))
			d[0:3,0]= dtmp[0:3,0]
			d[3,0]	= 1
			print d
			# 3D -> 2D
			f = T*d
			print 'f: ',f
			fx = f[0]
			fy = f[1]
			fz = f[2]
			fw = f[3]
			print 'fx: ',float(fx)
			print 'fy: ',float(fy)
			print 'fw: ',float(fw)
			if fw<>0.0: # TODO: why is fw always zeros? only in euqal plane scenario?
				print 'True'
				bx = fx/fw
				by = fy/fw
			else:
				bx = fx
				by = fy
			# convert meters to pixels
			bxp=bx/pitch
			byp=by/pitch
			print 'bx: ',bx
			print 'by: ',by
			print 'bxp: ',bxp
			print 'byp: ',byp
			X[yp,xp] = bxp # pixel coordinate on display
			Y[yp,xp] = byp # pixel coordinate on display

	height_out	= np.max(np.max(Y))
	width_out	= np.max(np.max(X))
	image_new	= np.zeros((height_out,width_out,channels),dtype=np.uint8)
	# perform forward mapping for testing
	# TODO: this must be a backward mapping if forward gives correct results
	for x in range(width):
		for y in range(height):
			# TODO: perform interpolation instead of rounding
			x_tmp = int(X[y,x])
			y_tmp = int(Y[y,x])
			#print 'x: ',x,'-> ',x_tmp
			#print 'y: ',y,'-> ',y_tmp
			#if x_tmp >= width or y_tmp >= height or x_tmp<0 or y_tmp < 0:
			#	image_new[y,x,:] = 0
			#else:
				#print 'Before'
				#print image[y,x,:]
				#print image_new[y,x,:]
			image_new[x_tmp,y_tmp,:] = copy.deepcopy(image[y,x,:])
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
	cv2.imshow(windowname,image)
	cv2.waitKey(1)
	windowname = 'test'
	cv2.namedWindow(windowname)
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
