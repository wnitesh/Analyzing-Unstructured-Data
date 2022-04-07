####Part1: Text Representation

import nltk
import pandas as pd
import numpy as np

#load data in a data frame
data = pd.read_csv("Assignment 1.csv", names = ['Number', 'Review'])

#Step1: Tokenize each review and save the output in a list within a list format
token_review_list = []
for n in data['Review']:
    token_each_review = nltk.word_tokenize(n)
    token_review_list.append(token_each_review)
print(token_review_list)

#Step2: Lemmatize each review and save the output in a list within a list format
lemma_token_review_list = []
lemmatizer = nltk.stem.WordNetLemmatizer()
for each_review in token_review_list:
    lemma_token_each_review = []
    for each_word in each_review:
        if each_word.isalpha():
            token = lemmatizer.lemmatize(each_word.lower()) #convert in lowercase and lemmatize
            lemma_token_each_review.append(token)
    lemma_token_review_list.append(lemma_token_each_review)
print(lemma_token_review_list)

#Step3: Remove Stop Words and save the output in a list within a list format
from nltk.corpus import stopwords
stop_words = stopwords.words("english")
stopwords_removed_list = []
for list in lemma_token_review_list:
    temp_stopwords_list = []
    for words in list:
        if words not in stop_words: #remove stopwords
            if words.isalpha(): #remove punctuations
                temp_stopwords_list.append(words)
    stopwords_removed_list.append(temp_stopwords_list)
print(stopwords_removed_list)
len(stopwords_removed_list)

#Step4: TFIDF Vectorization
#4.1: Merge individual words to create a single list containing all the reviews
reviews_list_post_stopwords = []
for temp_ind_review in stopwords_removed_list:
        reviews_list_post_stopwords.append(" ".join(temp_ind_review))
print(reviews_list_post_stopwords)

#4.2: TFIDF Vectorize
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=3)
vectorizer.fit(reviews_list_post_stopwords)
vectorized_output = vectorizer.transform(reviews_list_post_stopwords)

#4.3: Save ouput in csv
vectorized_array_output = vectorized_output.toarray()
print(vectorized_array_output)
my_array = np.array(vectorized_array_output)
df = pd.DataFrame(my_array)
df.to_csv("Part1_step4_output.csv")

#Step5: POS Tagging and TFIDF Vectorizing
#5.1: POS Tagging
POS_Review_list = []
for tokenized_doc in token_review_list:
    POS_token_doc = nltk.pos_tag(tokenized_doc)
    POS_token_temp = []
    for i in POS_token_doc:
        POS_token_temp.append(i[0] + i[1])
    POS_Review_list.append(" ".join(POS_token_temp))
print(POS_Review_list)
len(POS_Review_list)

#5.2: TFIDF Vectorize
vectorizer_POS = TfidfVectorizer(min_df=4)
vectorizer_POS.fit(POS_Review_list)
print(vectorizer_POS.vocabulary_)
len(vectorizer_POS.vocabulary_.keys())
POS_vectorized = vectorizer_POS.transform(POS_Review_list)

#5.3: Save output in csv
POS_vectorized_array_output = POS_vectorized.toarray()
print(POS_vectorized_array_output)
my_array = np.array(POS_vectorized_array_output)
df = pd.DataFrame(my_array)
df.to_csv("Part1_step5_output.csv")


#### Part 2: Word Embedding

#Step1: Index Based Encoding
#1.1: Create a list of all words
word_embedding_collection = token_review_list[0:10]
list_all_words = [j for i in word_embedding_collection for j in i]
print(list_all_words)

#1.2: Use sklearn package to encode
from numpy import array
from sklearn.preprocessing import LabelEncoder
index_encoder = LabelEncoder()
index_encoder = index_encoder.fit(list_all_words) # define vocabulary
index_encoded = [index_encoder.transform(doc) for doc in word_embedding_collection]
print(index_encoded)

#1.3: Save ouput in list format
index_encoded2l = [index_encoder.transform(doc).tolist() for doc in word_embedding_collection]
print(index_encoded2l)
#Get dimensions
for i in range(0,10):
    print(len(index_encoded2l[i]))

#1.4: Save output in csv
index_encoded2l_array=np.array(index_encoded2l,dtype=object)
np.savetxt("Part2_step1_output.csv", index_encoded2l_array, delimiter=",", fmt='%s')




#Step2: one hot encoding
#2.1: Create a list of all indexed words
indices_list = [[j] for i in index_encoded2l for j in i]
print(indices_list)

#2.2: One Hot Encode each indexed word
from sklearn.preprocessing import OneHotEncoder
onehot_encoder = OneHotEncoder(sparse=False)
onehot_encoder_fit = onehot_encoder.fit(indices_list)
onehot_encoded = [onehot_encoder_fit.transform([[i] for i in doc_i]).tolist() for doc_i in index_encoded]
print(onehot_encoded)
#Get Dimensions
for i in range(0,10):
    print(len(onehot_encoded[i]))

#2.3: Save the output file in csv format
output = []
for temp in onehot_encoded:
    for elem in temp:
        output.append(elem)

output_array=np.array(output,dtype=object)
np.savetxt("Part2_step2_output.csv", output_array, delimiter=",", fmt='%s')




#Part3: Use Pretrained model GloVe
#3.1: Convert GloVe to word2vec
from gensim.test.utils import get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
target_file = get_tmpfile('word2vec.6B.300d.txt')
glove2word2vec('glove.6B.50d.txt', target_file)

#3.2 Load GloVe model
from gensim.models import KeyedVectors
filename = 'glove.6B.50d.txt.word2vec'
model = KeyedVectors.load_word2vec_format(filename, binary=False)

#3.3: print GloVe vocabulary
words = model.key_to_index
print(words)

#3.4: Use pretrained model to get 50d index for restaurant reviews
glove_reviews_tensor = []
for each_doc in word_embedding_collection:
    glove_each_doc_vector = []
    for tokenized_words in each_doc:
        if tokenized_words in words.keys():
            glove_each_doc_vector.append(model[tokenized_words].tolist())
    glove_reviews_tensor.append(glove_each_doc_vector)
print(glove_reviews_tensor)
#Get Dimensions
for i in range(0,10):
    print(len(glove_reviews_tensor[i]))

#3.5: Save the output file in csv format
output_glove = []
for temp in glove_reviews_tensor:
    for elem in temp:
        output_glove.append(elem)
output_array=np.array(output_glove,dtype=object)
np.savetxt("Part2_step3_output.csv", output_array, delimiter=",", fmt='%s')












