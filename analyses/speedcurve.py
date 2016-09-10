#!/usr/bin/python

import os
import numpy as np
import cv

def main():
	# define paths
	dir			= 'media/results'
	sizefile	= os.path.join(dir,'sizes.csv')
	timefile	= os.path.join(dir,'times.csv')

	# load data
	sizedata = np.loadtxt(open(sizefile,"rb"),delimiter=",")
	timedata = np.loadtxt(open(timefile,"rb"),delimiter=",")

	print sizedata
	print timedata



if __name__ == "__main__":
	main()
