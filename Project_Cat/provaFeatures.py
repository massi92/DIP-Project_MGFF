import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import decomposition
from sklearn.cross_validation import cross_val_score
from sklearn import cross_validation
from FileManager import FileManager
from detectAndExtract import detectAndExtract
from provaHOG import provaHOG

"""
COME E' SUDDIVISO LO SCRIPT
1. Creazione oggetto obj della classe per la rilevazione delle FEATURES
2. Scelta del percorso delle immagini, della keywords positiva, dell'estensione.
3. Vengono creati due diversi vettori: 
	uno contiene la lista di tutte le immagini che vogliamo usare come esempi positivi (keywords),
	l'altro contiene lista di tutte le immagini per esempi negativi (no-keywords)
4. Viene chiamato il metodo elabora dell'obj per estrarre le features. Queste vengono salvate in un file csv (positivo e negativo) nella cartella della keywords
5. Vengono riletti i file Csv e salvati in un array
6. Viene costruito un file Target (0: positivo, 1: negativo).
7. Viene addestrato il modello con la cross_validation (Per prova RF)
8. Vengono mostrati a video gli errori del riconoscimento di esempi positivi e negativi
"""

# DETECTOR AND EXTRACTOR
detector = "HARRIS"
extractor = "SIFT"
obj = detectAndExtract(detector,extractor)
#obj = provaHOG()
#cut_features = 30000

## PREPROCESSING PERCORSI IMMAGINI
fm = FileManager()
root = "./imm/"
keywords = "fish"
estension = ".jpg"
verbose=False
file_positive = root+keywords+"/"+keywords+"_" + detector + "_" + extractor + "_positive.csv"
file_negative = root+keywords+"/"+keywords+"_" + detector + "_" + extractor + "_negative.csv"

print "Preprocessing Analisi Cartelle e Files"
# List dir contiene tutte le cartelle della root img
listDir = fm.listNoHiddenDir(root)

# Cerco l'indice della cartella della Keywords
indexKeywords = listDir.index(keywords)
# Vado ad eliminare nella lista cartelle quella della keywords
listDir.pop(indexKeywords)

# Creo una lista contenente tutte le immagini (jpg) positive (Filtro estension)
listArrayPositive = fm.listNoHiddenFiles(root+keywords,estension)

# Creo una lista contenente tutte le immagini negative (Filtro estension)
listArrayNegative = []
for i in range(0,len(listDir)):
	item = fm.listNoHiddenFiles(root+listDir[i],estension)
	listArrayNegative.append(item)

#print listDir
#print listArrayNegative

#raw_input()

## SAVE FEATURES IN FILES
# Controllo se e' gia' presente
if not os.path.isfile(file_positive):
	print "Nuovo File Features Positive " + detector + " + " + extractor + " creato."
	X_positive = []
	for k in range(0,len(listArrayPositive)): # Da eliminare file csv ed altri
		base_name = root+keywords+"/"+keywords+fm.correggi(k)+ estension
		print "Immagini " + str(k) + " -> " + base_name 
		ret = (obj.elabora(base_name)) #[0:cut_features]
		X_positive.append(ret)

	fm.arrayToCsv(X_positive,file_positive)
else:
	print "File Features Positive " + detector + " + " + extractor + " esiste gia."

if not os.path.isfile(file_negative):
	print "Nuovo File Features Negative " + detector + " + " + extractor + " creato."
	X_negative = []
	for i in range(0,len(listDir)):
		print "#################################"
		print "Elaborazione " + listDir[i]
		for k in range(0,len(listArrayNegative[i])):
			base_name = root + listDir[i] + "/" + listDir[i] + fm.correggi(k) + estension
			print "Immagini " + str(k) + " -> " + base_name 
			ret = (obj.elabora(base_name)) #[0:cut_features]
			X_negative.append(ret)

	fm.arrayToCsv(X_negative,file_negative)
else:
	print "File Features Negative " + detector + " + " + extractor + " esiste gia."


#raw_input()


## FEATURES ML FROM FILES
# Array dai file csv
print "Caricamento file csv into array"
X_positive_from_csv = fm.csvToArray(file_positive)
X_negative_from_csv = fm.csvToArray(file_negative)

X_total = X_positive_from_csv + X_negative_from_csv
y_total = [0]*len(X_positive_from_csv)+[1]*len(X_negative_from_csv)

#print y_total

# Preparing cross_validation
#X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_total, y_total, test_size=0.3, random_state=1)
X_train = X_total
y_train = y_total
X_test = [obj.elabora("./gatto.jpg")]
y_test = [1]

# ML
print "Machine Learning"
clf = RandomForestClassifier(n_estimators=300,verbose=verbose,n_jobs=3)
#clf = AdaBoostClassifier(base_estimator=None, n_estimators=50, learning_rate=1.0, algorithm='SAMME.R', random_state=None)
#clf = svm.OneClassSVM(kernel="rbf",gamma=2,degree=4)
#clf = GaussianNB()
clf = clf.fit(X_train, y_train)
#scores = cross_val_score(clf,X_train,y_train)
print "Indici di prestazioni"
#print "cross_val_score: " + str(scores.mean())
#print "test score: " + str(clf.score(X_test,y_test))

# Previsione
y_pred = clf.predict(X_test)
print clf.predict_proba(X_test)

print "La classe prevista e': " + str(y_pred) + " e quella corretta e': " + str(y_test)

""" Previsione Manuale """

countTotPositive = 0
countErrorPositive = 0
countTotNegative = 0
countErrorNegative = 0
for i in range(0,len(y_pred)):
	if y_test[i] == 0:
		countTotPositive +=1
		if y_pred[i] != y_test[i]:
			countErrorPositive +=1
	elif y_test[i] == 1:
		countTotNegative +=1
		if y_pred[i] != y_test[i]:
			countErrorNegative +=1

print "Totali campioni Testing: " + str(len(y_test))
print "Errore positivo (testing) di " + keywords + " e' di " + str(countErrorPositive) + " su " + str(countTotPositive)
print "Errore negativo (testing) di " + keywords + " e' di " + str(countErrorNegative) + " su " + str(countTotNegative)


"""
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.OrRd):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, [keywords, "not " + keywords], rotation=30)
    plt.yticks(tick_marks, [keywords, "not " + keywords])
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)
print('Confusion matrix, without normalization')
print(cm)
plt.figure()
plot_confusion_matrix(cm)
plt.show()
"""

#raw_input()