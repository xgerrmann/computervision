#!/usr/bin/python

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys

def main(argv):
	# define paths
	dir			= 'media/results'
	normal = True;
	if(len(argv)==1):
		datafile	= os.path.join(dir,'comp_distribution.csv')
	elif(argv[1]=="normal"):
		datafile	= os.path.join(dir,'comp_distribution.csv')
	elif(argv[1]=="skip_homography"):
		datafile	= os.path.join(dir,'comp_distribution_skip_hom.csv')
		normal = False;
	else:
		print "-------------------"
		print "Usage:\n./analyses/comp_distro normal\nor:\n./analyses/comp_distro skip_homography"
		print "-------------------"

	# load data
	data = np.loadtxt(open(datafile,"rb"),delimiter=",")

	# convert from seconds to milliseconds
	data *= 1000
	

# IMPORTANT: skip first row of data, the GPU startup is of big influence

	# show data
	plt.figure(facecolor='white')
	plt.boxplot(data[1:,:-1],notch=False,sym='',)
	plt.title('Computational time for each program section')
	if normal == True:
		plt.ylim([0,50])
	else:
		plt.ylim([0,150])
	#plt.xlim([min(size),max(size)])
	plt.ylabel("Time [ms]")
	labels = ['','Webcam', 'Headpose', 'Reset Image', 'Manage Pose', 'Homography', 'Display']
	plt.xticks(range(8),labels,rotation=-45)
	plt.tight_layout()
	#plt.show()
	
	total_mean = data[1:,-1].mean(axis=0);
	total_std  = data[1:,-1].std(axis=0);
	print data[1:,-1]
	print 'Mean:' + str(total_mean);
	print 'Std: ' + str(total_std);
	print 'Frame rate:' + str(1/(total_mean/1000));

	# save plot
	if(normal):
		plt.savefig('media/plots/comp_distro.eps',format='eps', facecolor="white")
	else:
		plt.savefig('media/plots/comp_distro_skip_hom.eps',format='eps', facecolor="white")

if __name__ == "__main__":
	main(sys.argv)
