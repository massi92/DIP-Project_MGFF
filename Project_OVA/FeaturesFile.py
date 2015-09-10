import numpy as np
import cv2
import os

from FileManager import FileManager
from detectAndExtract import detectAndExtract
from HOG import HOG

class FeaturesFile(object):
	""" Permette di creare file di features """
	detector = None
	extractor = None
	keywords = None
	root = None
	estension = None

	obj = None

	def __init__(self, detector, extractor, keywords, root=os.path.dirname(__file__)+os.path.sep+"imm"+os.path.sep, estension=".jpg"):
	#def __init__(self, detector, extractor, keywords, root="./imm2/", estension=".jpg"):		
		super(FeaturesFile, self).__init__()
		self.detector = detector
		self.extractor = extractor
		self.keywords = keywords
		self.root = root
		self.estension = estension
		if detector== "HOG" and extractor == "HOG":
			self.obj = HOG()
		else:
			self.obj = detectAndExtract(detector,extractor)

	def creaFileFeatures(self,key=None):

		if key == None:
			## PREPROCESSING PERCORSI IMMAGINI
			fm = FileManager()
			file_positive = self.root+self.keywords+"/"+self.keywords+"_" + self.detector + "_" + self.extractor + ".csv"

			# Creo una lista contenente tutte le immagini (jpg) positive (Filtro estension)
			listArrayPositive = fm.listNoHiddenFiles(self.root+self.keywords,self.estension)

			## SAVE FEATURES IN FILES
			# Controllo se e' gia' presente, altrimenti leggo il file e restituisco array
			if not os.path.isfile(file_positive):
				print "Nuovo File Features Positive " + self.detector + " + " + self.extractor + " creato."
				X_positive = []
				for k in range(0,len(listArrayPositive)): # Da eliminare file csv ed altri
					base_name = self.root+self.keywords+"/"+self.keywords+fm.correggi(k)+ self.estension
					print "Immagini " + str(k) + " -> " + base_name 
					ret = (self.obj.elabora(base_name)) #[0:cut_features]
					X_positive.append(ret)

				fm.arrayToCsv(X_positive,file_positive)
			else:
				print "File Features Positive " + self.detector + " + " + self.extractor + " esiste gia."
				X_positive = fm.csvToArray(file_positive)
		else:
			## PREPROCESSING PERCORSI IMMAGINI
			fm = FileManager()
			file_positive = self.root+key+"/"+key+"_" + self.detector + "_" + self.extractor + ".csv"

			# Creo una lista contenente tutte le immagini (jpg) positive (Filtro estension)
			listArrayPositive = fm.listNoHiddenFiles(self.root+key,self.estension)

			## SAVE FEATURES IN FILES
			# Controllo se e' gia' presente, altrimenti leggo il file e restituisco array
			if not os.path.isfile(file_positive):
				print "Nuovo File Features Positive " + self.detector + " + " + self.extractor + " creato."
				X_positive = []
				for k in range(0,len(listArrayPositive)): # Da eliminare file csv ed altri
					base_name = self.root+key+"/"+key+fm.correggi(k)+ self.estension
					print "Immagini " + str(k) + " -> " + base_name 
					ret = (self.obj.elabora(base_name)) #[0:cut_features]
					X_positive.append(ret)

				fm.arrayToCsv(X_positive,file_positive)
			else:
				print "File Features Positive " + self.detector + " + " + self.extractor + " esiste gia."
				X_positive = fm.csvToArray(file_positive)

		return X_positive

	def negativeFeatures(self):
		## PREPROCESSING PERCORSI IMMAGINI
		fm = FileManager()
		#file_negative = self.root+self.keywords+"/"+self.keywords+"_" + self.detector + "_" + self.extractor + "_negative.csv"

		# List dir contiene tutte le cartelle della root img
		listDir = fm.listNoHiddenDir(self.root)

		# Cerco l'indice della cartella della Keywords
		indexKeywords = listDir.index(self.keywords)
		# Vado ad eliminare nella lista cartelle quella della keywords
		listDir.pop(indexKeywords)

		X_negative = []
		for i in range(0,len(listDir)):
			file_negative = self.root+listDir[i]+"/"+listDir[i]+"_" + self.detector + "_" + self.extractor + ".csv"
			if not os.path.isfile(file_negative):
				self.creaFileFeatures(listDir[i])
			else:
				X = fm.csvToArray(file_negative)
				X_negative = X_negative + X
		return X_negative

	def negativeFeaturesSingleCat(self,category):
		## PREPROCESSING PERCORSI IMMAGINI
		fm = FileManager()

		X_negative = []
		#for i in range(0,len(listDir)):
		file_negative = self.root+category+"/"+category+"_" + self.detector + "_" + self.extractor + ".csv"
		if not os.path.isfile(file_negative):
			self.creaFileFeatures(category)
		else:
			X = fm.csvToArray(file_negative)
			X_negative = X_negative + X
		return X_negative		

	def getFeatures(self,category):
		X_positive = self.creaFileFeatures()
		X_negative = self.negativeFeatures()
		#X_negative = self.negativeFeaturesSingleCat(category)
		return X_positive, X_negative

	def getObj(self):
		return self.obj

	def getKeyword(self):
		return self.keywords

if __name__ == '__main__':
	ff = FeaturesFile("HARRIS","SIFT","cat")
	ff.creaFileFeatures()
