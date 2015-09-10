import numpy as np
import cv2
import os
from preprocess import Preprocess

class HOG(object):
	"""classe per fare HOG"""
	#def __init__(self, arg):
	#	super(provaHOG, self).__init__()
	#	self.arg = arg

	def elabora(self,path):
		#print path
		img = cv2.imread(path,0)
		
		#Trasformazione
		pp = Preprocess()
		img = pp.applyTransform(img,300,300)

		descriptorValues = []
		locations = []
		#hd = cv2.HOGDescriptor((32,64), (16,16), (8,8), (8,8), 9)
		#hd = cv2.HOGDescriptor()
		hd = cv2.HOGDescriptor((16,16), (16,16), (8,8), (8,8),9)
		#hd = cv2.HOGDescriptor()
		#print "Lunghezza di hd: "+str(len(hd))
		res = hd.compute(img)		
		#ls = res.tolist()
		#print str(len(ls))
		"""
		des = res[0]

		for i in xrange(1,len(res)):
			des = np.concatenate([des,res[i]])
		"""
		return res.ravel()
		#print ls
		#print str(len(res))
		#for i in range(0,len(ls)):
		#	des = des+ls[i]
		print "#####################"	
		#return des


		#return des.tolist()
		#print des

if __name__ == '__main__':
	os.system('cls')

	obj = HOG()

	vt = obj.elabora('./imm/cat/cat003.jpg')
	vt1 = obj.elabora('./imm/cat/cat004.jpg')

	#print cv2.CalcEMD2(vt,vt1)
	#print vt
	print str(len(vt))

		