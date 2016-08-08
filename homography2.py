#!/usr/bin/python
# homography.py
# script to test and perform homographies on an example image
# script based on: http://math.stackexchange.com/questions/494238/how-to-compute-homography-matrix-h-from-corresponding-points-2d-2d-planar-homog/1886060#1886060
import cv2
import numpy as np
from paths import default_image
import math
import sys
import copy

def performhomography(windowname,image):
	pitch = 0.2625*(10**-3) # [m] pixel pitch (pixel size) assume square pixels, which is generally true
	print 'Pitch: %.6f'%pitch
	
	rx = 0.0*math.pi
	ry = 0.0*math.pi
	rz = 0.0*math.pi

	(height,width,channels)	= image.shape
	(Rx,Ry,Rz)		= calcrotationmatrix(rx,ry,rz)
	Rt	= Rz*Ry*Rx
	Rti	= np.linalg.inv(Rt)
	# define 3 points on the virtual image plane
	p0 = np.matrix([[0],[0],[0]])
	p1 = np.matrix([[1],[0],[0]])
	p2 = np.matrix([[0],[1],[0]])
	# preform rotation of points
	pr0 = Rt*p0
	pr1 = Rt*p1
	pr2 = Rt*p2
	# find 2 vectors that splan the plane:
	# pr0 is always <0,0,0>, so the vectors pr1 and pr2 define the plane

	# construct the vectors for the view-lines from the optical center to the corners of the virtual image:
	# Corner numbering:
	# 0-----1
	# |		|
	# 3-----2
	# pixels
	# corner [x,y]
	cp0	= np.matrix([[0],[0]])
	cp1	= np.matrix([[width],[0]])
	cp2	= np.matrix([[width],[height]])
	cp3	= np.matrix([[0],[height]])
	corners_p = [cp0, cp1,cp2,cp3]
	print corners_p
	# meters
	f	= 0.3 # [m]
	e	= np.matrix([[0],[0],[-f]]) # position of eye
	wx	= width*pitch	# [m]
	hy	= height*pitch	# [m]
	c0	= np.matrix([[-wx/2],[+hy/2],[0]])-e # vector from eye to corner 0
	c1	= np.matrix([[+wx/2],[+hy/2],[0]])-e # vector from eye to corner 1
	c2	= np.matrix([[+wx/2],[-hy/2],[0]])-e # vector from eye to corner 2
	c3	= np.matrix([[-wx/2],[-hy/2],[0]])-e # vector from eye to corner 3
	cornerlines = [c0,c1,c2,c3]
	print 'Lines:\n',cornerlines
	# For each intersection a linear combination of the vectors spanning the plane exists
	# when this combination is found, the exact location of the intersection is known
	corners_proj = []
	for ic, c in enumerate(cornerlines):
		# find the projection of each corner point on the plane
		# note: origin is still center of the plane
		A	= np.hstack((c,-pr1,-pr2))
		Ai	= np.linalg.inv(A)
		print A
		print Ai
		#print 'Corner: \n',c
		comb = Ai*e
		#a[ic] = res[0]
		#b[ic] = res[1]
		intersection = A[:,1:]*comb[1:]
		#print 'Intersection:\n', intersection
		# compute x,y coordinates in plane by performing the inverse plane rotation on the point of intersection
		coords = Rti*intersection
		#print 'Coordinates:\n',coords
		x = (coords[0]+wx/2)/pitch
		y = -(coords[1]-hy/2)/pitch # change y direction
		#print x
		#print y
		corners_proj.append([float(x),float(y)])
		#corners_proj[0,ic]=x
		#corners_proj[1,ic]=y
	print corners_proj
	
	# temporary
	# corners [x,y]
	#corners_proj = ([0,0],[50,0],[50,50],[0,50])
	print corners_proj

	# corners in the projection are now known.
	# calculate the homography
	# source: http://math.stackexchange.com/questions/494238/how-to-compute-homography-matrix-h-from-corresponding-points-2d-2d-planar-homog
	M1 = np.zeros((0,8))
	M2 = np.zeros((8,1))
	for i in range(4):
		xA	= float(corners_p[i][0])
		yA	= float(corners_p[i][1])
		xB	= float(corners_proj[i][0])
		yB	= float(corners_proj[i][1])
		print 'xA: ',xA,', xB: ',xB
		print 'yA: ',yA,', yB: ',yB
		row0	= np.matrix([xA,yA,1,0,0,0,-xA*xB,-yA*xB])
		row1	= np.matrix([0,0,0,xA,yA,1,-xA*yB,-yA*yB])
		M1 = np.vstack((M1,row0))
		M1 = np.vstack((M1,row1))
		M2[i*2,0]	= xB
		M2[i*2+1,0]	= yB
	print M1
	print M2
	H = np.linalg.inv(np.transpose(M1)*M1)*(np.transpose(M1)*M2)
	print H
	H = np.vstack((H,[1]))
	H = np.reshape(H,(3,3))
	Hi = np.linalg.inv(H)
	print H

	# apply homography backward
	image_out = np.zeros((height,width,channels),dtype=np.uint8)
	for h in range(height):
		for w in range(width):
			tmp = np.matrix([[w],[h],[1]])
			res = Hi*tmp
			scale	= res[2]
			xtmp	= int(res[0]/scale)
			ytmp	= int(res[1]/scale)
			if xtmp<0 or xtmp>=width:
				xtmp=0
			if ytmp<0 or ytmp>=height:
				ytmp=0
			#print 'y:',h,'\t->',ytmp
			#print 'x:',w,'\t->',xtmp
			image_out[h,w,:] = image[ytmp,xtmp,:]
			#print image_out[h,w,:]
			#print image[ytmp,xtmp,:]
			if h <> ytmp or w <> xtmp:
				#print 'weird >>'
				#sys.exit('strange')
				pass
	cv2.imshow('test',image_out)
	cv2.waitKey(0)


def calcrotationmatrix(rx, ry, rz):
	# source: http://nghiaho.com/?page_id=846
	# source: https://en.wikipedia.org/wiki/3D_projection (uses negative angles?)
#	rx = -rx
#	ry = -ry
#	rz = -rz
	Rx	= np.matrix([[1, 0, 0],
					[0,	math.cos(rx), -math.sin(rx)],
					[0, math.sin(rx), math.cos(rx)]])
	Ry= np.matrix([[math.cos(ry), 0, math.sin(ry)],
					[0, 1, 0],
					[-math.sin(ry), 0, math.cos(ry)]])
	Rz	= np.matrix([[math.cos(rz), -math.sin(rz), 0],
					[math.sin(rz), math.cos(rz), 0],
					[0, 0, 1]])
	return (Rx,Ry,Rz)

def main():
	image		= cv2.imread(default_image)
	windowname	= "image"
	cv2.namedWindow(windowname)
	cv2.imshow(windowname,image)
	cv2.waitKey(100)
	windowname = 'test'
	cv2.namedWindow(windowname)
	
	performhomography(windowname, image)

	return 0

if __name__ == '__main__':
	main()
