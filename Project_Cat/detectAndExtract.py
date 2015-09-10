import numpy as np
import cv2
import os
from preprocess import Preprocess

"""
Questa Classe contiene il metodo "elabora" che estrae dei detector e da tali detector estrai descriptor per le features delle immagini.
"""

"""
Supported Feature Detector Algorithms
FAST FastFeatureDetector
STAR StarFeatureDetector
SIFT SIFT (nonfree module)
SURF SURF (nonfree module)
ORB ORB
BRISK BRISK
MSER MSER
GFTT GoodFeaturesToTrackDetector
HARRIS GoodFeaturesToTrackDetector with Harris detector enabled
Dense DenseFeatureDetector
SimpleBlob SimpleBlobDetector


Supported Descriptor Extractor Algorithms

SIFT SIFT
SURF SURF
BRIEF BriefDescriptorExtractor
BRISK BRISK
ORB ORB
FREAK FREAK       

"""
class detectAndExtract:
	_detector = None
	_descriptor = None
	_useKpAsAttr = False

	def __init__(self,detector,descriptor,flag=False):
		#super(detectAndExtract, self).__init__()
		self._detector = detector
		self._descriptor = descriptor
		self._useKpAsAttr = flag
			
	def elabora(self,path):
		img1 = cv2.imread(path,0)  	
		detectorObj = cv2.FeatureDetector_create(self._detector)
		if self._detector == "SIFT":
			detectorObj.setInt("nFeatures",1000)
		elif self._detector == "SURF":
			detectorObj.setInt("hessianThreshold",4000)
		
		descriptorExtractor = cv2.DescriptorExtractor_create(self._descriptor)
		
		keypoints = detectorObj.detect(img1)
		
		(keypoints, descriptors) = descriptorExtractor.compute(img1, keypoints)
		#print keypoints
		#print descriptors
		#raw_input()

		"""
		print self._detector+" parameters (dict):", detectorObj.getParams()
		for param in detectorObj.getParams():
			ptype = detectorObj.paramType(param)
			if ptype == 0:
				print param, "=", detectorObj.getInt(param)
			elif ptype == 2:
				print param, "=", detectorObj.getDouble(param)
		"""
		

		out = []

		if self._useKpAsAttr == False:
			out = descriptors[0]

			for i in range(0,len(descriptors)):
				np.concatenate([out,descriptors[i]])	
			
			return out.tolist()
		else:
			#uso dei kp come singole istanze del classificatori
			#return descriptors.tolist()
			return descriptors

if __name__ == '__main__':
	obj = detectAndExtract("SIFT","SIFT",True)
	ar = obj.elabora("./imm/ball/ball000.jpg")
	#print ar
	#print "###############"
	#for i in range(0,len(ar)):
	#	print len(ar[i])