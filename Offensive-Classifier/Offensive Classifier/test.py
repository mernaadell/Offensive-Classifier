import csv
from Preprocessor import preprocess
import pickle
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report


with open('trainedModel','rb') as training_model:
	model = pickle.load(training_model)

testDataFile = open('trial-data/offenseval-trial.txt')
testDataTSV = csv.reader(testDataFile, dialect = 'excel-tab')

X = []
Y = []
NOT = 0
OFF = 0
testData = list(testDataTSV)
for i in range(1,len(testData)):
	X.append(testData[i][0])
	Y.append(testData[i][1])
	if Y[-1] == 'NOT':
		NOT += 1
	else:
		OFF += 1

# print(NOT)
# print(OFF)
# exit()
documents = preprocess(X)

vectorizer = pickle.load(open("vectorizer.pk","rb"))
selector = pickle.load(open("selector.pk","rb"))
X = selector.transform(vectorizer.transform(documents).toarray())

y_pred = model.predict(X)
print("Test\nConfusion matrix\n")
print(confusion_matrix(Y, y_pred))  
print("Classification report\n")
print(classification_report(Y, y_pred)) 
print("\nAccuarcy:",accuracy_score(Y,y_pred))
