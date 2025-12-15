# %%
import numpy as np
import pandas as pd
import nltk
import string

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import gensim
from gensim.models import Word2Vec, KeyedVectors

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

# %%
#text preprocessing
reviews = pd.read_csv("imdbReviews.csv")

# %%
reviews.head()

# %%
reviews['Sentiment'].value_counts()

# %%
#save the labels and encode them as 1 and 0 for future classification/clustering
from sklearn.preprocessing import LabelEncoder

enc = LabelEncoder()
label = enc.fit_transform(reviews['Sentiment'])
print(label[:10])
print(reviews['Sentiment'][:10])

# %%
#change the text column datatype to string
reviews = reviews.astype({'Text':'string'})

# %%
#get the review text for preprocessing
text = reviews['Text']

# %%
text1 = []

for review in text:
    #print(sentence)
    #remove punctuation
    review = review.translate(str.maketrans('', '', string.punctuation))
    # remove digits/numbers
    review = review.translate(str.maketrans('', '', string.digits))
    #change to lowercase
    review = review.lower()
    #print(sentence)
    text1.append(review)

# %%
text1 = pd.Series(text1)

# %%
#remove stop words

#Setting English stopwords
stop_words = set(stopwords.words('english'))

text1 = text1.apply(lambda x: ' '.join(term for term in x.split() if term not in stop_words))
text1[:5]

# %%
#apply stemming
ps = nltk.PorterStemmer()

text1 = text1.apply(lambda x: ' '.join(ps.stem(term) for term in x.split()))

# %%
reviews1 = list(zip(text1, label))

# %%
reviewsP = pd.DataFrame (reviews1, columns = ['Review', 'Sentiment'])

# %%
reviewsP1 = reviewsP.sample(frac=1, random_state=1).reset_index()

# %%
#split the dataset

#train dataset by splitting the data
train_reviews = reviewsP1.Review[:1400]
train_sentiments = reviewsP1.Sentiment[:1400]

#test dataset
test_reviews = reviewsP1.Review[1400:]
test_sentiments = reviewsP1.Sentiment[1400:]

print(train_reviews.shape,train_sentiments.shape)
print(test_reviews.shape,test_sentiments.shape)

# %%
train_reviews.head()

# %%
#tokenise the data
tokenized_reviews = train_reviews.apply(lambda x: x.split())

# %%
#learn vectors from the data
model = gensim.models.Word2Vec(
            tokenized_reviews,
            vector_size=100, # desired no. of features/independent variables
            window=5,        # context window size
            min_count=2,     # Ignores all words with total frequency lower than 2.
            sg = 1,          # 1 for skip-gram model
            hs = 0,
            negative = 10,   # for negative sampling
            workers= 32,     # no.of cores
            seed = 34
)


# %%
model.train(tokenized_reviews, total_examples= len(train_reviews), epochs=20)

# %%
embeddingsSize=100

def getVectors(dataset):
  singleDataItemEmbedding=np.zeros(embeddingsSize)
  vectors=[]
  for dataItem in dataset:
    wordCount=0
    for word in dataItem:
      if word in model.wv:
        singleDataItemEmbedding=singleDataItemEmbedding+model.wv.key_to_index[word]
        wordCount=wordCount+1

    singleDataItemEmbedding=singleDataItemEmbedding/wordCount
    vectors.append(singleDataItemEmbedding)
  return vectors

# %%
trainReviewVectors=getVectors(train_reviews)
testReviewVectors=getVectors(test_reviews)

# %%
############################################
###           Decision Tree              ###
############################################
#training the model
DT=DecisionTreeClassifier(criterion ='entropy', random_state= 0)

DT=DT.fit(trainReviewVectors,train_sentiments)

DT_predict=DT.predict(testReviewVectors)


DT_report=classification_report(test_sentiments,DT_predict,target_names=['Positive','Negative'])
print(confusion_matrix(test_sentiments,DT_predict), '\n')
print(DT_report)

# %%



