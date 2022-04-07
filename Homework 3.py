## Part 1

import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

df = pd.read_excel("Assignment 3.xlsx", index_col='id')

test = pd.concat((df.iloc[400:500,:],df.iloc[900:1000,:]),axis=0)
train = pd.concat((df.iloc[0:400,:], df.iloc[500:900, :]),axis=0)
x_train = train[['review']]
y_train = train[['label']]
x_test = test[['review']]
y_test = test[['label']]

x_train.shape

#Document Preprocessing
token_list=[]
for i in x_train['review']:
    token_list.append(nltk.word_tokenize(i.lower()))
lammetize_list=[]
for i in token_list:
    lemmatizer=nltk.stem.WordNetLemmatizer()
    lemmatized_token=[lemmatizer.lemmatize(token) for token in i]
    lammetize_list.append(lemmatized_token)
stop_list=[]
for i in lammetize_list:
    stop_words_removed = [token for token in i if not token in stopwords.words('english') if token.isalpha()]
    stop_list.append(stop_words_removed)

#Join words back to string
def listToString(s):
    str1 = ""
    for ele in s:
        str1 = str1+" "+ele
    return str1

final_list=[]
for i in stop_list:
    sentence_list=listToString(i)
    final_list.append(sentence_list)

#TFIDF Vectorizer
vectorizer=TfidfVectorizer(min_df=5,ngram_range=(1,2))
v1=vectorizer.fit(final_list)
v2=vectorizer.transform(final_list)
x_train=pd.DataFrame(v2.toarray(),columns=v1.vocabulary_.keys())

#Test Data

token_list_test=[]
for i in x_test['review']:
    token_list_test.append(nltk.word_tokenize(i.lower()))
lammetize_list_test=[]
for i in token_list_test:
    lemmatizer=nltk.stem.WordNetLemmatizer()
    lemmatized_token=[lemmatizer.lemmatize(token) for token in i]
    lammetize_list_test.append(lemmatized_token)
stop_list_test=[]
for i in lammetize_list_test:
    stop_words_removed = [token for token in i if not token in stopwords.words('english') if token.isalpha()]
    stop_list_test.append(stop_words_removed)

final_list_test=[]
for i in stop_list_test:
    sentence_list=listToString(i)
    final_list_test.append(sentence_list)
#Changing it w.r.t Tfidf vector
v_test=vectorizer.transform(final_list_test)
x_test=pd.DataFrame(v_test.toarray(),columns=v1.vocabulary_.keys())

#Models
#Naive Bayes

from sklearn.metrics import accuracy_score
## Naive Bayes
from sklearn.naive_bayes import MultinomialNB
NBmodel = MultinomialNB()
# training
NBmodel.fit(x_train, y_train)
y_pred_NB = NBmodel.predict(x_test)
# evaluation
acc_NB = accuracy_score(y_test, y_pred_NB)
print("Naive Bayes model Accuracy:: {:.4f}%".format(acc_NB*100))

#logit
from sklearn.linear_model import LogisticRegression
Logitmodel = LogisticRegression()
# training
Logitmodel.fit(x_train, y_train)
y_pred_logit = Logitmodel.predict(x_test)
# evaluation
from sklearn.metrics import accuracy_score
acc_logit = accuracy_score(y_test, y_pred_logit)
print("Logit model Accuracy:: {:.4f}%".format(acc_logit*100))

#RandomForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
RFmodel = RandomForestClassifier(n_estimators=100, max_depth=15, bootstrap=True, random_state=0) ## number of trees and number of layers/depth
#training
RFmodel.fit(x_train, y_train)
y_pred_RF = RFmodel.predict(x_test)
#evaluation
acc_RF = accuracy_score(y_test, y_pred_RF)
print("Random Forest Model Accuracy: {:.4f}%".format(acc_RF*100))

#SVM
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
SVMmodel = LinearSVC()
# training
SVMmodel.fit(x_train, y_train)
y_pred_SVM = SVMmodel.predict(x_test)
# evaluation
acc_SVM = accuracy_score(y_test, y_pred_SVM)
print("SVM model Accuracy: {:.2f}%".format(acc_SVM*100))

#ANN
from sklearn.neural_network import MLPClassifier
DLmodel = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(4,), random_state=1)
# training
DLmodel.fit(x_train, y_train)
y_pred_DL= DLmodel.predict(x_test)
# evaluation
acc_DL = accuracy_score(y_test, y_pred_DL)
print("DL model Accuracy: {:.4f}%".format(acc_DL*100))


#LSTM
import pandas as pd
nltk.download('punkt')
df = pd.read_excel("Assignment 3.xlsx", index_col='id')
test = pd.concat((df.iloc[400:500,:],df.iloc[900:1000]),axis=0)
train = pd.concat((df.iloc[0:400,:], df.iloc[500:900, :]),axis=0)
x_train = train[['review']]
y_train = train[['label']]
x_test = test[['review']]
y_test = test[['label']]

#encoding each label and transforming both train and test datasets
import nltk
from numpy import array
from sklearn.preprocessing import LabelEncoder
tokenized_list= [nltk.word_tokenize(doc.lower()) for doc in df['review']]
tokenized_list_test= [nltk.word_tokenize(doc.lower()) for doc in x_test['review']]
tokenized_list_train= [nltk.word_tokenize(doc.lower()) for doc in x_train['review']]

# A set for all possible words
words = [j for i in tokenized_list for j in i]
total_words=len(words)
index_encoder = LabelEncoder()
index_encoder = index_encoder.fit(words) # define vocabulary
x_train_lstm = [index_encoder.transform(doc) for doc in tokenized_list_train]
x_test_lstm = [index_encoder.transform(doc) for doc in tokenized_list_test]

#dummy encoding y labes
import numpy as np
y_train_lstm = np.array([1 if label =='movie' else 0 for label in y_train['label']])
y_test_lstm = np.array([1 if label =='movie' else 0 for label in y_test['label']])

#padding
from keras.preprocessing import sequence
maxlen = 100
x_train_lstm = sequence.pad_sequences(x_train_lstm, maxlen=maxlen)
x_test_lstm = sequence.pad_sequences(x_test_lstm, maxlen=maxlen)

# build model
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten, Dropout
from keras.layers import LSTM
max_features = total_words
batch_size = 100

# model architecture
model = Sequential()
model.add(Embedding(max_features, 20, input_length=maxlen))
model.add(LSTM(40, dropout=0.20, recurrent_dropout=0.20))
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train_lstm, y_train_lstm, batch_size=batch_size, epochs=10, validation_data=(x_test_lstm, y_test_lstm))


## Part 2

import keras
from keras import utils
from keras.datasets import cifar10
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

#train/test split
(x_train_cnn, y_train_cnn), (x_test_cnn, y_test_cnn) = cifar10.load_data()

#visualize first 20 datapoints (images)
import matplotlib.pyplot as plt
# load dataset
(trainX, trainy), (testX, testy) = cifar10.load_data()
rows =4
columns = 5
# define figure
fig=plt.figure(figsize=(10, 10))
# visualize these random images
for i in range(1, columns*rows +1):
    fig.add_subplot(rows, columns, i)
    plt.imshow(testX[i-1])
    plt.xticks([])
    plt.yticks([])
    plt.title("{}".format(testy[i-1]))
plt.show()

# bc the expected y is binary class matrices
y_train_cnn = utils.np_utils.to_categorical(y_train_cnn, 10)
y_test_cnn = utils.np_utils.to_categorical(y_test_cnn, 10)

#CNN Model building
CNNmodel=Sequential()
CNNmodel.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
CNNmodel.add(Dropout(0.2))
CNNmodel.add(Conv2D(32, (3, 3), activation='relu'))
CNNmodel.add(MaxPooling2D(pool_size=(2, 2)))
CNNmodel.add(Conv2D(64, (3, 3), activation='relu'))
CNNmodel.add(Dropout(0.2))
CNNmodel.add(Conv2D(64, (3, 3), activation='relu'))
CNNmodel.add(MaxPooling2D(pool_size=(2, 2)))
CNNmodel.add(Flatten())
CNNmodel.add(Dense(256, activation='relu'))
CNNmodel.add(Dropout(0.2))
CNNmodel.add(Dense(10, activation='softmax'))
CNNmodel.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

##Model Fit
CNNmodel.fit(x_train_cnn, y_train_cnn, validation_data=(x_test_cnn, y_test_cnn), batch_size=500, epochs=5)

##Model Performance
performance = CNNmodel.evaluate(x_test_cnn, y_test_cnn)
print('Test accuracy:', performance[1])

#Additional Layers
CNNmodel=Sequential()
CNNmodel.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
CNNmodel.add(Dropout(0.2))
CNNmodel.add(Conv2D(32, (3, 3), activation='relu'))
CNNmodel.add(MaxPooling2D(pool_size=(2, 2)))
CNNmodel.add(Conv2D(64, (3, 3), activation='relu'))
CNNmodel.add(Dropout(0.2))
CNNmodel.add(Conv2D(64, (3, 3), activation='relu'))
CNNmodel.add(MaxPooling2D(pool_size=(2, 2)))
CNNmodel.add(Conv2D(128, (3, 3), activation='relu'))
CNNmodel.add(Dropout(0.2))
CNNmodel.add(Conv2D(128, (3, 3), activation='relu'))
CNNmodel.add(Flatten())
CNNmodel.add(Dense(256, activation='relu'))
CNNmodel.add(Dropout(0.2))
CNNmodel.add(Dense(10, activation='softmax'))
CNNmodel.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

## model fit
CNNmodel.fit(x_train_cnn, y_train_cnn, validation_data=(x_test_cnn, y_test_cnn), batch_size=500, epochs=5)

##model performance
performance = CNNmodel.evaluate(x_test_cnn, y_test_cnn)
print('Test accuracy:', performance[1])

#increase Eopchs to 20
CNNmodel=Sequential()
CNNmodel.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
CNNmodel.add(Dropout(0.2))
CNNmodel.add(Conv2D(32, (3, 3), activation='relu'))
CNNmodel.add(MaxPooling2D(pool_size=(2, 2)))
CNNmodel.add(Conv2D(64, (3, 3), activation='relu'))
CNNmodel.add(Dropout(0.2))
CNNmodel.add(Conv2D(64, (3, 3), activation='relu'))
CNNmodel.add(MaxPooling2D(pool_size=(2, 2)))
CNNmodel.add(Conv2D(128, (3, 3), activation='relu'))
CNNmodel.add(Dropout(0.2))
CNNmodel.add(Conv2D(128, (3, 3), activation='relu'))
CNNmodel.add(Flatten())
CNNmodel.add(Dense(256, activation='relu'))
CNNmodel.add(Dropout(0.2))
CNNmodel.add(Dense(10, activation='softmax'))
CNNmodel.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

## model fit
CNNmodel.fit(x_train_cnn, y_train_cnn, validation_data=(x_test_cnn, y_test_cnn), batch_size=500, epochs=20)

##Model Performance
performance = CNNmodel.evaluate(x_test_cnn, y_test_cnn)
print('Test accuracy:', performance[1])

#Classic ML models. Naïve Bayes and Random Forest
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
#Create Empty List
flatten_x_train, flatten_x_test = ([] for i in range(2))
#Flatten the input image
for image in x_train:
  x = image.flatten()
  flatten_x_train.append(x)
for image in x_test:
  x = image.flatten()
  flatten_x_test.append(x)

#Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
gnb = GaussianNB()
y_pred = gnb.fit(flatten_x_train, y_train).predict(flatten_x_test)
accuracy = accuracy_score(y_test, y_pred)
print("accuracy score", accuracy)

#Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
RFmodel = RandomForestClassifier(n_estimators=100, max_depth=10, bootstrap=True, random_state=0) ## number of trees and number of layers/depth
#Model Training
RFmodel.fit(flatten_x_train, y_train)
y_pred_RF = RFmodel.predict(flatten_x_test)
#Evaluation
acc_RF = accuracy_score(y_test, y_pred_RF)
print("Random Forest Model Accuracy: {:.4f}%".format(acc_RF*100))
