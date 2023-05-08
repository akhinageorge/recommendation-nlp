import pandas as pd
import gensim
import gensim.downloader as api
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

wv=api.load('word2vec-google-news-300')

df = pd.read_csv('subjectsPOC.csv', header=0).dropna()
subjects_list = df['Word'].tolist()

subjects_vectors={}
for subject in subjects_list:
    try:
        subjects_vectors[subject] = wv[subject]
    except KeyError:
        pass    

input_subject = 'Hindi'
input_vector = wv.get_vector(input_subject)

similarities = {}
for subject, vector in subjects_vectors.items():
    sim = cosine_similarity([input_vector], [vector])[0][0]
    similarities[subject] = sim
similar_subjects = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:5]

print(similar_subjects)
