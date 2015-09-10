import os
import sys

path = os.path.dirname(__file__)

#os.path.dirname(__file__).join("\\Project_KPCat")
#os.path.dirname(__file__).join("\\Project_Cat")
#os.path.dirname(__file__).join("\\Project_OVA")

print path

def printMain():
	print "DIP PROJECT - IMAGE CLASSIFIER"
	print "Choose the option you want to use: "
	print "1 - Use KP as single row of the classifier"
	print "2 - Use an image as a single row of the classifier"
	print "3 - Use the OneVsAll Classifier"
	print "\n"
	opt = input("Insert the option: ")

	if opt>=1 and opt<=3:
		if opt==1:
			newPath = path+os.path.sep+"Project_KPCat"
			sys.path.insert(0, newPath)
			execfile(newPath+os.path.sep+"main.py")
		elif opt==2:
			newPath = path+os.path.sep+"Project_Cat"
			sys.path.insert(0, newPath)
			execfile(newPath+os.path.sep+"main.py")
		elif opt==3:
			newPath = path+os.path.sep+"Project_OVA"
			sys.path.insert(0, newPath)
			execfile(newPath+os.path.sep+"main.py")
	else:
		print "Insert a correct option value!"


if __name__ == '__main__':
	printMain()