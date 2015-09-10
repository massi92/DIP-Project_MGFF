import numpy as np
import cv2
import os

class Preprocess(object):
	""" Classe con operazioni di preprocessing su immagini """

	def adaHistEq(self,img):
		clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
		cl1 = clahe.apply(img)	
		return cl1

	def removeNoise(self,img):
		kernel = np.ones((5,5),np.float32)/25	#kernel
		dst = cv2.filter2D(img,-1,kernel)
		return dst

	def changeSize(self,img,width,height):
		res = cv2.resize(img,(width,height))
		return res

	def toBin(self,img):
		th = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
		return th

	# Da importare pySaliencyMap
	def saliency(self,img):		
	    imgsize = img.shape
	    img_width  = imgsize[1]
	    img_height = imgsize[0]
	    sm = pySaliencyMap.pySaliencyMap(img_width, img_height)
	    map = sm.SMGetSM(img)
	    return map	

	def applyTransform(self,img,width,height):
		out1 = self.changeSize(img,width,height)
		#out1 = self.toBin(out1)
		out1 = self.adaHistEq(out1)
		#out1 = self.removeNoise(img)	
		
		return out1	
	
if __name__ == '__main__':
	#Prova

	path = './imm/cat/cat000.jpg'
	img = cv2.imread(path,0)

	pp = Preprocess()

	#out1 = pp.applyTransform(img,500,500)
	out1 = pp.toBin(img)


	cv2.imshow('Immagine trasformata',out1)
	cv2.waitKey(0)