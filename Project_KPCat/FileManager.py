import numpy as np
import os
import csv


class FileManager(object):
	""" Classe per gestire i file """
	def listNoHiddenFiles(self,root,extension=""):
		array = []
		for item in os.listdir(root):
			if not item.startswith('.') and os.path.isfile(os.path.join(root,item)):
				if item.endswith(extension):
					array.append(item)
		return array

	def listNoHiddenFilesAndDir(self,root):
		array = []
		for item in os.listdir(root):
			if not item.startswith('.'):
				array.append(item)
		return array

	def listNoHiddenDir(self,root):
		array = []
		for item in os.listdir(root):
			if not item.startswith('.') and not os.path.isfile(os.path.join(root,item)):
				array.append(item)
		return array

	def arrayToCsv(self,array,name):
		"""
		Array = np.array(array)
		print Array
		np.savetxt(name, array, delimiter=",")
		"""
		for i in range(0,len(array)):	
			Array = np.array(array)		
			np.savetxt(name, array[i], delimiter=",")		

	def csvToArray(self,name):
		with open(name, 'rb') as f:
			reader = csv.reader(f)
			array = list(reader)
		#print len(array)
		#for i in range(0,len(array)):
		#	print len(array[i])
		#raw_input()
		return array

	def correggi(self,value):
		if(value<10):
			return "00"+str(value)
		else:
			return "0"+str(value)


if __name__ == '__main__':
	root = "./img/bottle/"
	fm = FileManager()
	
	print "###### File and Dirs"
	NoHiddenFilesAndDir = fm.listNoHiddenFilesAndDir(root)
	print len(NoHiddenFilesAndDir)

	print "###### Dirs"
	NoHiddenDir = fm.listNoHiddenDir(root)
	print len(NoHiddenDir)

	print "###### Files"
	NoHiddenFiles = fm.listNoHiddenFiles(root)
	print len(NoHiddenFiles)

	# print "# Scrivo su files"
	# fm.arrayToCsv([[0,1],[2,3]],"array.csv")