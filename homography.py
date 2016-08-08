#!/usr/bin/python
# homography.py
# script to test and perform homographies on an example image

import cv2
import numpy as np
from paths import default_image
import math
import sys
import copy

## Sources
# Example:		http://dlib.net/face_landmark_detection.py.html
# Speeding up:	http://www.learnopencv.com/speeding-up-dlib-facial-landmark-detector/

def calcrotationmatrix(rx, ry, rz):
	# source: http://nghiaho.com/?page_id=846
	# source: https://en.wikipedia.org/wiki/3D_projection (uses negative angles?)
	rx = -rx
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
	#R	= Z*Y*X
	R	= X*Y*Z
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

def calctransmatrix(e):
	ex = e[0]
	ey = e[1]
	ez = e[2]

	# T: transformation matrix for projection of 3D -> 2D
	T = np.zeros((4,4))
	T[0,0] = 1
	T[1,1] = 1
	T[2,2] = 1
	T[0,2] = -ex/ez
	T[1,2] = -ey/ez
	T[3,2] = 1/ez
	for i in range(4):
		for j in range(4):
			print '%6.6f'%float(T[i,j])
	return T

def performhomography(windowname,image):
	rx	= 0.0*math.pi
	ry	= 0.0*math.pi
	rz	= 0.0*math.pi
	R	= calcrotationmatrix(rx, ry, rz)
	
	pitch = 0.2625*(10**-3) # [m] pixel pitch (pixel size) assume square pixels, which is generally true
	print 'Pitch: %.6f'%pitch
	
#	tx	= 0 # translation x
#	ty	= 0 # translation y
#	tz	= 0 # translation z

	f = -1 # TODO: must be detected accurately for correct results
	
	# e: viewers position relative to the camera
	ex = 0
	ey = 0
	ez = -f
	e = np.matrix([[ex],[ey],[ez]]) # viewer is direcly behind projection plane

	T = calctransmatrix(e)

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
			# also set origin to center of image
			x = (xp-width_out/2)*pitch
			y = (yp-height_out/2)*pitch
			#a = np.matrix([[x],[y],[0]]) # virtual image is always at z=0
			a = np.matrix([[x],[y],[0.5]]) # virtual image is always at z=0
			
			print 'a: ',a
			# c is the 3D position of the camera
			c = np.matrix([[0],[0],[1]])
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
			#print 'f: ',f
			fx = f[0]
			fy = f[1]
			fz = f[2]
			fw = f[3]
			print 'fx: ',float(fx)
			print 'fy: ',float(fy)
			print 'fw: ',float(fw)
			if fw<>0.0: # TODO: why is fw always zeros? only in equal plane scenario?
			#	print 'True'
				sys.exit('Weird')
				bx = fx/fw
				by = fy/fw
			else:
				bx = fx
				by = fy
				#print 'bx0: ',bx
				#print 'bx1: ',-f/d[2,0]*d[0,0]-0
			# convert meters to pixels
			bxp=bx/pitch+width_out/2
			byp=by/pitch+height_out/2
			#print 'bx: ',bx
			#print 'by: ',by
			#print 'bxp: ',bxp
			#print 'byp: ',byp
			X[yp,xp] = bxp # pixel coordinate on display
			Y[yp,xp] = byp # pixel coordinate on display

	height_out	= int(np.max(np.max(abs(Y))))
	width_out	= int(np.max(np.max(abs(X))))
	print height_out
	print width_out
	image_new	= np.zeros((height_out+1,width_out+1,channels),dtype=np.uint8)
	print image_new.shape
	# perform forward mapping for testing
	# TODO: this must be a backward mapping if forward gives correct results
	for x in range(width):
		for y in range(height):
			# TODO: perform interpolation instead of rounding
			x_tmp = int(X[y,x])
			y_tmp = int(Y[y,x])
			#print 'x: ',x,'\t-> ',x_tmp
			#print 'y: ',y,'\t-> ',y_tmp
			#if x_tmp >= width or y_tmp >= height or x_tmp<0 or y_tmp < 0:
			#	image_new[y,x,:] = 0
			#else:
				#print 'Before'
				#print image[y,x,:]
				#print image_new[y,x,:]
			#x_tmp = abs(x_tmp)
			#y_tmp = abs(y_tmp)
			x_tmp = x_tmp
			y_tmp = y_tmp
			image_new[y_tmp,x_tmp,:] = copy.deepcopy(image[y,x,:])
				#print 'After'
				#print image[y,x,:]
				#print image_new[y,x,:]

				#image_new = image
	cv2.imshow('test',image_new)
	cv2.waitKey(2000)
	return 0

def main():
	image		= cv2.imread(default_image)
	windowname	= "image"
	cv2.namedWindow(windowname)
	cv2.imshow(windowname,image)
	cv2.waitKey(10)
	windowname = 'test'
	cv2.namedWindow(windowname)
	
	performhomography(windowname, image)

	return 0

if __name__ == '__main__':
	main()
