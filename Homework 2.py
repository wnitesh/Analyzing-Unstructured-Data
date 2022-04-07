from PIL import Image
from pylab import *

##Part 1: Image data image normalization

#Step 1: Read the image corpus and resize into 100x100 pixels
image_corpus = ['1.png','2.png','3.png','4.png','5.png','6.png','7.png','8.png','9.png','10.png']

image_corpus_open = []
for image in image_corpus:
    temp = Image.open(image)
    figure()
    imshow(temp)
    image_corpus_open.append(temp)

image_corpus_resize = []
for image2 in image_corpus_open:
    temp2=image2.resize((100,100))
    figure()
    imshow(temp2)
    image_corpus_resize.append(temp2)

#Step2: Convert the image corpus into greyscale
image_corpus_greyscale = []
for image3 in image_corpus_resize:
    temp3 = array(image3.convert('L'))
    figure()
    imshow(temp3)
    image_corpus_greyscale.append(temp3)

#Step3.1: Flatten the grayscale corpus into 1D vector and draw histogram
image_corpus_greyscale_flatten = []
for image4 in image_corpus_greyscale:
    temp4 = image4.flatten() #flatten the corpus
    figure()
    hist(temp4,256) #draw histogram for each corpus
    image_corpus_greyscale_flatten.append(temp4)

#Step3.2: Export the flatten file
output_array=np.array(image_corpus_greyscale_flatten,dtype=object)
np.savetxt("Part1_Step3_output.csv", output_array, delimiter=",", fmt='%s')

#Step4: Perform histogram equalization
for image5 in image_corpus_greyscale_flatten:
    imhist, bins = histogram(image5, 256, normed=True)
    cdf = imhist.cumsum()
    cdf = 255 * cdf / cdf[-1] #normalization
    temp5 = interp(image5, bins[:-1], cdf)
    figure()
    hist(temp5.flatten(),256)


## Part 2  LDA Topic Modelling

#Step1: Load and Transform the data
import nltk
import pandas as pd
df = pd.read_excel('Assignment 2 text.xlsx', index_col='id')
all_reviews = []
for i in df['review']:
    all_reviews.append(i)
token_reviews = []
for n in all_reviews:
    temp = nltk.word_tokenize(n) #Tokenize the reviews
    token_reviews.append(temp)

#Step1.2: Lemmatize each review and save the output in a list within a list format
lemma_token_review_list = []
lemmatizer = nltk.stem.WordNetLemmatizer()
for each_review in token_reviews:
    lemma_token_each_review = []
    for each_word in each_review:
        if each_word.isalpha():
            token = lemmatizer.lemmatize(each_word.lower())
            lemma_token_each_review.append(token)
    lemma_token_review_list.append(lemma_token_each_review)

#Step1.3: Join back sentences
reviews_list_post_lemma = []
for temp_ind_review in lemma_token_review_list:
    reviews_list_post_lemma.append(" ".join(temp_ind_review))
print(reviews_list_post_lemma)

#Step1.4: Create Term Document Matrix
from sklearn.feature_extraction.text import CountVectorizer
vectorizer1 = CountVectorizer(stop_words='english', min_df=5, ngram_range=(1,2))
X2 = vectorizer1.fit_transform(reviews_list_post_lemma)
terms = vectorizer1.get_feature_names()
terms

#Step2: LDA Topic Modelling for 6 Models
from sklearn.decomposition import LatentDirichletAllocation
lda = LatentDirichletAllocation(n_components=6).fit(X2)
for id, topic in enumerate(lda.components_):
    print("Topic %d:" % (id),topic)

#Step3: Topic Distribution for 10 restaurant and movie reviews
for id, review in enumerate(X2[0:10]):
    print('Review ' + str(id + 1))
    print('Topic Distribution ', lda.transform(review))
    print('Top-2 topics', lda.transform(review).argsort()[:,:-2-1:-1])

for id, review in enumerate(X2[500:510]):
    print('Review ' + str(id + 501))
    print('Topic Distribution ', lda.transform(review))
    print('Top-2 topics', lda.transform(review).argsort()[:,:-2-1:-1])

#Step4: Top 5 terms for each of 6 topics
for id, topic in enumerate(lda.components_):
    print("Topic %d:" % (id))
    print(" ".join([terms[i] for i in topic.argsort()[:-5-1:-1]]))

