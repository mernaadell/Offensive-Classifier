import csv
from Preprocessor import preprocess
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import SelectFromModel
from sklearn import tree
from sklearn import svm
import pickle
from sklearn.metrics import confusion_matrix,classification_report,f1_score

###### Reading dataset ######
trainingDataFile = open('training-v1/offenseval-training-v1.tsv')
trainingDataTSV = csv.reader(trainingDataFile, dialect = 'excel-tab')
X = []
Y = []
trainingData = list(trainingDataTSV)
for i in range(1,len(trainingData)):
	X.append(trainingData[i][1])
	Y.append(trainingData[i][2])

#### text preprocessing ####
documents = preprocess(X)

########## Vectorization ###########
tfidfVectorizer = TfidfVectorizer(max_features = 1600,min_df = 5,max_df = 0.7,stop_words = stopwords.words('english'))
X = tfidfVectorizer.fit_transform(documents).toarray()
with open('vectorizer.pk', 'wb') as fin:
	pickle.dump(tfidfVectorizer, fin)

############ Classification #############
print("Select classifier")
print("_________________\n")
print("1)Logistic regression")
print("2)Naive bayes")
print("3)Desicion tree")
print("4)SVM\n")
no = int(input("choose number: "))
if no >=1 and no <=4:
	print("wait a moment...\n")
else:
	print("Invalid input,please enter number from 1 to 4")
	exit()
if no == 1:
	min_max = MinMaxScaler()
	X = min_max.fit_transform(X)
	select = SelectFromModel(LogisticRegression(solver='sag',random_state = 0,C=2))
	model = LogisticRegression(solver ='sag',random_state=0,C=2)
elif no == 2:
	select = SelectFromModel(LogisticRegression(solver='sag',random_state = 0,C=2))
	model = MultinomialNB(alpha =1e-5)
elif no == 3:
	select=SelectFromModel(tree.DecisionTreeClassifier())   
	model = make_pipeline(StandardScaler(),tree.DecisionTreeClassifier(class_weight=None,
                                                                       criterion='gini', max_depth=15,
               max_features=None, max_leaf_nodes=None, min_samples_leaf=30,
              min_samples_split=15, min_weight_fraction_leaf=0.0,
               presort=False, random_state=100, splitter='best') ) 
else:
	min_max = MinMaxScaler()
	X = min_max.fit_transform(X)
	select=SelectFromModel(svm.SVC(kernel = 'linear'))   
	model = svm.SVC(kernel = 'linear')


select.fit(X,Y) 
x_select=select.transform(X) 
pickle.dump(select, open("selector.pk", "wb"))
X_train, X_cross_validation , Y_train , Y_cross_validation = train_test_split(x_select,Y,test_size = 0.2 , random_state = 0)
model.fit(X_train,Y_train)
picklefile = open('trainedModel','wb')
pickle.dump(model,picklefile)


Y_pred = model.predict(X_train)
print("Training\nConfusion matrix\n")
print(confusion_matrix(Y_train, Y_pred))  
print("Classification report\n")
print(classification_report(Y_train, Y_pred)) 
print("\nf1 score:",f1_score(Y_train,Y_pred, average='micro'))

Y_pred = model.predict(X_cross_validation)
print("\nCross validation\nConfusion matrix\n")
print(confusion_matrix(Y_cross_validation, Y_pred))  
print("Classification report\n")
print(classification_report(Y_cross_validation, Y_pred))  
print("\nf1 score:",f1_score(Y_cross_validation,Y_pred, average='micro'))