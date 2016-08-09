#!/usr/bin/python
# This script serves the purpose of finding how how the
# tensordut function from python really works and can be used for
# calculating the perspective warp of an image, given the (3x3) homography matrix

# M is the resulting matrix of dimension [height, width, 3]
# H is the homography matrix of size [3,3]
# O is the coordinate matrix with the X, Y and W (=scale) stacked in the 3rd dimension.
# X, Y and W are of dimension [height, width]

# for i,j,k:
# M[i,j,k]	=	H[k,:]*O[i,j,:]
# Is same as:
# for i,j,k,p:
# M[i,j,k]	+=	H[k,p]*O[i,j,p]
# The above formula should be implemented with the tensordot function instead
# of in a loop

import numpy as np
# numpy.stack requires at least numpy 1.10

X = np.array([[0,1],[0,1]])	# x coordinates
Y = np.array([[0,0],[1,1]])	# y coordinates
W = np.array([[1,1],[1,1]])	# scale is 1 everywhere
print 'X:\n',X
print 'Y:\n',Y
print 'W:\n',W
print Y
print X.shape
print Y.shape
O = np.stack((X,Y,W),2)

H = np.matrix([[1,0,0],[0,1,0],[0,0,1]]) # eye
print O.shape
print H.shape

#M = np.tensordot(H,O,axes=([0,1],[0,1,0]))

## Einsum expample
A = np.array([0, 1, 2])

B = np.array([[ 0,  1,  2,  3],
              [ 4,  5,  6,  7],
              [ 8,  9, 10, 11]])

T = A[:, np.newaxis]
print T*B
print (A[:, np.newaxis] * B).sum(axis=1)
print np.einsum('i,ij->i', A, B)

res = np.einsum('kp,ijp->ijk',H,O)
print res
print res.shape
print res[0,0,:]
print res[0,1,:]
print res[1,1,:]
print res[1,0,:]

# This gives correct results: np.einsum('kp,ijp->ijk',H,O)
