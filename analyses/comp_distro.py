#!/usr/bin/python

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

def main():
	# define paths
	dir			= 'media/results'
	datafile	= os.path.join(dir,'comp_distribution.csv')

	# load data
	data = np.loadtxt(open(datafile,"rb"),delimiter=",")

	# convert from seconds to milliseconds
	data *= 1000
	
	# show data
	plt.figure(facecolor='white')
	plt.boxplot(data[:,:-1],notch=False,sym='',)
	plt.title('Computational time vs number of pixels')
	plt.ylim([0,50])
	#plt.xlim([min(size),max(size)])
	plt.ylabel("Time [ms]")
	labels = ['','Webcam', 'Headpose', 'Reset Image', 'Manage Pose', 'Homography', 'Display']
	plt.xticks(range(8),labels,rotation=-45)
	plt.tight_layout()
	#plt.show()

	# save plot
	plt.savefig('media/plots/comp_distro.eps',format='eps', facecolor="white")
if __name__ == "__main__":
	main()
