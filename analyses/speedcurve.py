#!/usr/bin/python

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

def main():
	# define paths
	dir			= 'media/results'
	sizefile	= os.path.join(dir,'sizes.csv')
	timefile	= os.path.join(dir,'times.csv')

	# load data
	sizedata = np.loadtxt(open(sizefile,"rb"),delimiter=",")
	timedata = np.loadtxt(open(timefile,"rb"),delimiter=",")

	timedata = 1000*timedata
	
	# calculate mean and standard deviation
	mean_time = timedata.mean(axis=0);
	stdev_time = timedata.std(axis=0);
	
	size	= [elem[0]*elem[1] for elem in sizedata]
	
	t_max	= mean_time+stdev_time
	t_min	= mean_time-stdev_time
	# show data
	plt.figure(facecolor='white')
	plt.fill_between(size, t_min, t_max, facecolor='blue', interpolate=True,alpha = 0.5)
	plt.semilogx(size,mean_time, color='blue',linewidth=1)
	plt.title('Computational time vs number of pixels')
	plt.ylim([0,1])
	plt.xlim([min(size),max(size)])
	plt.xlabel("Size [pixels]")
	plt.ylabel("Time [ms]")
	#plt.show()

	# save plot
	plt.savefig('media/plots/size-time.png', facecolor="white")
if __name__ == "__main__":
	main()
