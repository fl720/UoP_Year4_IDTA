# %%
import numpy as np 
import pandas as pd 

import nltk
import string

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

import gensim
from gensim.utils import simple_preprocess

# %%
#text preprocessing (same as last week's lab)
reviews = pd.read_csv("imdbReviews.csv")

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

# %%
#apply stemming
ps = nltk.PorterStemmer()

text1 = text1.apply(lambda x: ' '.join(ps.stem(term) for term in x.split()))
text1[:5]

# %%
reviews1 = list(zip(text1, label))

# %%
reviewsP = pd.DataFrame (reviews1, columns = ['Review', 'Sentiment'])
reviewsP

# %%
data = reviewsP.Review.values.tolist()

# %%
data

# %%
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence)))

words = list(sent_to_words(data))

# %%
print(words[:1][0][:30])

# %%
import gensim.corpora as corpora                      

# Create Dictionary
id2word = corpora.Dictionary(words)                   

# Create Corpus
texts = words                                        

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]    

# View
print(corpus[:1][0][:30])

# %%
from pprint import pprint

# number of topics
num_topics = 10

# Build LDA model
lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                       id2word=id2word,
                                       num_topics=num_topics)

# Print the Keyword in the 10 topics
pprint(lda_model.print_topics())
#doc_lda = lda_model[corpus]

# %%
pip install pyldavis

# %%
import pyLDAvis.gensim_models
import pyLDAvis

# Visualize the topics
pyLDAvis.enable_notebook()

LDAvis = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)

LDAvis
# by sliding the lambda value, we can see how the term relevance changes. 1 means osme frequent terms that are also can be seem in other topics, while 0 means the terms that are more unique to the topic.

# %%



