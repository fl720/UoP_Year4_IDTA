# %%
import numpy as np               
import pandas as pd 
import nltk
import string

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


# %%
# - This package is large and takes some time to download, take the lighter version below.
# nltk.download()
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')

# %%
reviews = pd.read_csv("imdbReviews.csv")

# %%
reviews.head()

# %%
reviews.shape

# %%
reviews['Sentiment'].value_counts()

# %%
#save the labels and encode them as 1 and 0 for future classification/clustering
from sklearn.preprocessing import LabelEncoder

enc = LabelEncoder()
label = enc.fit_transform(reviews['Sentiment'])
print(label[:10])
print(reviews['Sentiment'][:10])

# %% [markdown]
# Preprocess the text and observe its transformation.

# %%
#change the text column datatype to string
reviews = reviews.astype({'Text':'string'})

# %%
reviews.dtypes

# %%
#get the review text for preprocessing
text = reviews['Text']
text[:5]

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
    
 
text1[:5]

# %%
text1 = pd.Series(text1)
text1[:5]

# %%
#remove stop words
nltk.download('stopwords')    
#Setting English stopwords
stop_words = set(stopwords.words('english'))

text1 = text1.apply(lambda x: ' '.join(term for term in x.split() if term not in stop_words))
text1[:5]

# %%
#apply lemmatising
from nltk.stem import WordNetLemmatizer 
nltk.download('wordnet')

# Init the Wordnet Lemmatizer
lemmatizer = WordNetLemmatizer()
text2 = text1.apply(lambda x:' '.join(lemmatizer.lemmatize(w) for w in x.split()))
text2[:5] #notice that it does not do a good job

# %%
#apply lemmatising with POS tags

from nltk.corpus import wordnet

def get_wordnet_pos(word):
    #Map POS tag to first character lemmatize() accepts
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

# Init the Wordnet Lemmatizer
lemmatizer = WordNetLemmatizer()
text3 = text1.apply(lambda x:' '.join(lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in x.split()))
text3[:5] #notice that it does a better job

# %%
#apply stemming
ps = nltk.PorterStemmer()

text4 = text1.apply(lambda x: ' '.join(ps.stem(term) for term in x.split()))
text4[:5]

# %%
reviews1 = list(zip(text3, label))

reviewsP = pd.DataFrame (reviews1, columns = ['Review', 'Sentiment'])
reviewsP

# %%
#transform text into BoW with count features
cv=CountVectorizer()

#cv_reviews 
cv_reviews = cv.fit_transform(reviewsP['Review'])

#see the features
cv.get_feature_names_out()

# %%
print(cv_reviews)

# %%
cv_reviews.shape

# %%
#to see the data in the typical tabular format
df =  pd.DataFrame(cv_reviews.toarray(), columns=cv.get_feature_names_out())
df.head()

# %%
#to filter by tf value and keep a certain number of features, use the max_features parameter
cv=CountVectorizer(max_features=10000)

#cv_reviews 
cv_reviews = cv.fit_transform(reviewsP['Review'])

cv_reviews.shape

# %%
# Trasform text into Tfidf representations
tv=TfidfVectorizer()

#transformed train reviews
tv_reviews=tv.fit_transform(reviewsP['Review'])

print(tv.get_feature_names_out())

# %%
tv_reviews.shape

# %%
print(tv_reviews[0])


# %%
#to filter by tf value and keep a certain number of features, use the max_features parameter
tv=TfidfVectorizer(max_features=10000)

#transformed train reviews
tv_reviews=tv.fit_transform(reviewsP['Review'])

tv_reviews.shape

# %%
#get to top n features with the highest tf-idf 
feature_names = np.array(tv.get_feature_names_out())

def get_top_tf_idf_words(tv_reviews, top_n=10):
    importance = np.argsort(np.asarray(tv_reviews.sum(axis=0)).ravel())[::-1]
    return feature_names[importance[:top_n]] 

print([get_top_tf_idf_words(tv_reviews,10)])

# %%
df1 =  pd.DataFrame(tv_reviews.toarray(), columns=tv.get_feature_names_out())

df1['movie'].describe()

# %%
df1['movie'].hist()

# %%
df1['film'].describe()

# %%



