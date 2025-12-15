# %%
!pip install bertopic

# %%
import numpy as np
import pandas as pd
import nltk
import string

# %%
reviews = pd.read_csv("imdbReviews.csv")

# %%
reviews.head()

# %%
# Create a new column containing the length each review
reviews["Text_len"] = reviews["Text"].apply(lambda x : len(x.split()))

# %%
print("The longest review has: {} words".format(reviews.Text_len.max()))

# %%
# Visualize the length distribution
import seaborn as sns
import matplotlib.pyplot as plt
sns.displot(reviews.Text_len, kde=False)

# %%
text = reviews['Text']

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


text1[:2]

# %%
text1 = pd.Series(text1)
text1[:2]

# %%
#remove stop words
nltk.download('stopwords')

from nltk.corpus import stopwords

stopwords = nltk.corpus.stopwords.words('english')
newStopWords = ['film','movie', 'get', 'see', 'make', 'one']
stopwords.extend(newStopWords)

#Setting English stopwords
stop_words = set(stopwords)

#text = reviews['Text']
text2 = text1.apply(lambda x: ' '.join(term for term in x.split() if term not in stop_words))
text2[:5]

# %%
#apply lemmatising with POS tags

nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer


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
text3 = text2.apply(lambda x:' '.join(lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in x.split()))
text3[:2]

# %%
reviews['Text'] = text3
reviews.head()

# %%
# Create a new column containing the length each review after preprocessing
reviews["Text_len"] = reviews["Text"].apply(lambda x : len(x.split()))

# %%
# Visualize the length distribution
import seaborn as sns
import matplotlib.pyplot as plt
sns.displot(reviews.Text_len, kde=False)

# %%
from bertopic import BERTopic

#default uses HDBSCAN
model = BERTopic(verbose=True,embedding_model='paraphrase-MiniLM-L3-v2', top_n_words=10, min_topic_size= 20)
review_topics, probs = model.fit_transform(reviews.Text)

# %%
freq = model.get_topic_info()
print("Number of topics: {}".format( len(freq)))
freq.head()

# %%
freq.head(9)

# %%
a_topic = freq.iloc[1]["Topic"] # Select the 1st topic
model.get_topic(a_topic) # Show the words and their c-TF-IDF scores; the c-TF-IDF score is TF-IDF of the term in the cluster

# %%
model.visualize_barchart(n_words=10)

# %%
model.visualize_topics()

# %%
#Bert topic with k-means

from sklearn.cluster import KMeans

cluster_model = KMeans(n_clusters=10)
#KMmodel = BERTopic(hdbscan_model=cluster_model)

KMmodel = BERTopic(hdbscan_model=cluster_model, verbose=True, embedding_model='paraphrase-MiniLM-L3-v2', top_n_words=10, min_topic_size= 20)
KM_topics, KMprobs = KMmodel.fit_transform(reviews.Text)

# %%
KMfreq = KMmodel.get_topic_info()
print("Number of topics: {}".format( len(KMfreq)))
KMfreq.head()

# %%
KMfreq.head(10)

# %%
a_topic = freq.iloc[1]["Topic"] # Select the 1st topic
KMmodel.get_topic(a_topic) # Show the words and their c-TF-IDF scores; the c-TF-IDF score is TF-IDF of the term in the cluster

# %%
KMmodel.visualize_barchart(top_n_topics=10, n_words = 10)

# %%
KMmodel.visualize_topics()

# %%
from sentence_transformers import SentenceTransformer
from umap import UMAP

# Prepare embeddings
docs = reviews.Text
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = sentence_model.encode(docs, show_progress_bar=False)

# Train BERTopic
topic_model = BERTopic().fit(docs, embeddings)

# Run the visualization with the original embeddings
topic_model.visualize_documents(docs, embeddings=embeddings)

# Reduce dimensionality of embeddings, this step is optional but much faster to perform iteratively:
#reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
#topic_model.visualize_documents(docs, reduced_embeddings=reduced_embeddings)


