import emoji
import re
import nltk
from nltk.stem.porter import PorterStemmer

	
def preprocess(X):
	documents = []

	for i in range(0,len(X)):
		#remove @USER
		document = str(X[i]).replace("@USER"," ")
		document = emoji.demojize(document)
		document = str(document).replace(":"," ")
		#remove any not english character
		document = re.sub("[^A-Za-z ]", "", document)

		#remove single characters
		document = re.sub(r'\s+[a-zA-z]\s+', ' ' , document)

		#substituting multiple spaces with single space
		document = re.sub(r'\s+',' ',document,flags = re.I)

		#converting to lowercase
		document = document.lower()

		#Stemming
		document = nltk.word_tokenize(document)
		stemmer = PorterStemmer()
		document = [stemmer.stem(word) for word in document]
		document = ' '.join(document)
		
		documents.append(document)

	return documents