t# %%
import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

import nltk
import string

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

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
# as we can see, positive = 1 and negative = 0

# %%
#change the text column datatype to string
reviews = reviews.astype({'Text':'string'})

# %%
#get the review text for preprocessing
text = reviews['Text']
#text[:5]

# %%
text1 = []

# --- text preprocessing ---
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
    
 
#text1[:5]

# %%
text1 = pd.Series(text1)
#text1[:5]

# %%
#remove stop words
    
#Setting English stopwords
stop_words = set(stopwords.words('english'))

text1 = text1.apply(lambda x: ' '.join(term for term in x.split() if term not in stop_words))
#text1[:5]

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
#ttransform the data using tf-idf with a maximum of 10000 features
tv=TfidfVectorizer(max_features=10000)

#transformed train reviews
tv_reviews=tv.fit_transform(reviewsP['Review'])

tv_reviews.shape

# %%
#######################################
###             k-means             ###
#######################################

kmeans = KMeans(n_clusters=2)
kmeans.fit(tv_reviews)

#clusters = kmeans.labels_.tolist()

labels=kmeans.labels_
clusters=pd.DataFrame(list(zip(text,labels)),columns=['title','cluster'])

for i in range(2):
        print(clusters[clusters['cluster'] == i])

# %%
#get to top n features with the highest tf-idf 
feature_names = np.array(tv.get_feature_names_out())

def get_top_tf_idf_words(tv_reviews, top_n=10):
    importance = np.argsort(np.asarray(tv_reviews.sum(axis=0)).ravel())[::-1]
    return feature_names[importance[:top_n]] 

print([get_top_tf_idf_words(tv_reviews,10)])

# %%
df =  pd.DataFrame(tv_reviews.toarray(), columns=tv.get_feature_names_out())

# %%
#visualising the clusters
fig, axes = plt.subplots(1, 2, figsize=(16,8))

fte_colors = {0: "#008fd5", 1: "#fc4f30"}
   
km_colors = [fte_colors[label] for label in kmeans.labels_]

axes[0].scatter(df['movi'], df['time'], c=km_colors)
axes[1].scatter(df['movi'], df['time'], c=kmeans.labels_, cmap=plt.cm.Set1)
axes[0].set_title('Actual', fontsize=18)
axes[1].set_title('K_Means', fontsize=18)

# %%
#append dataframe with cluster number
df['cluster'] = kmeans.labels_

# %%
#view proterties of each cluster
cluster1=df.query("cluster == 0")
cluster2=df.query("cluster == 1")

# %%
cluster1.shape

# %%
cluster2.shape

# %%
cluster1_mean = kmeans.cluster_centers_[0]
cluster2_mean = kmeans.cluster_centers_[1]

# %%
len(cluster1.keys())

# %%
#visualised the first 10 features for the 2 clusters
cluster1_mean_p1 = cluster1_mean[:10]
cluster2_mean_p1 = cluster2_mean[:10]

X = cluster1.keys()[:10]

  
X_axis = np.arange(len(X))
  
plt.bar(X_axis - 0.2, cluster1_mean_p1, 0.4, label = 'Cluster1')
plt.bar(X_axis + 0.2, cluster2_mean_p1, 0.4, label = 'Cluster2')
  
plt.xticks(X_axis, X, rotation='vertical')
plt.xlabel("Groups")
#plt.subplots_adjust(bottom=0.1)
plt.ylabel("Mean values")
plt.title("Mean values per attribute")
plt.legend()
plt.show()

# %%
######################################################
###             Hierarchical Clustering            ###
######################################################
from sklearn.metrics.pairwise import cosine_similarity

dist = 1 - cosine_similarity(tv_reviews)

# %%
from scipy.cluster.hierarchy import ward, dendrogram

linkage_matrix = ward(dist) #define the linkage_matrix using ward clustering pre-computed distances

fig, ax = plt.subplots(figsize=(15, 20)) # set size
ax = dendrogram(linkage_matrix, orientation="right");

plt.tick_params(\
    axis= 'x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off')

plt.tight_layout() #show plot with tight layout

# %%



