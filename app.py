import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.title("Movie Recommender System")   

df = pd.read_csv("cleaned_data1.csv")   

# Create vectors from movie tags
cv = CountVectorizer(max_features=5000, stop_words="english")
vectors = cv.fit_transform(df["tags"]).toarray()

# Calculate cosine similarity between all movies
similarities = cosine_similarity(vectors)


movies = df['title'].tolist()

name = st.selectbox("Select a movie", movies)

#Lets write function get name of movie by index
def get_name_by_index(i):
    if 0 <= i < len(df):
        return df.loc[i,'title']
    else:
        return''

def get_index_from_name(name):
    # Normalize user input: lowercase it and strip out all spaces and hyphens
    clean_user_name = name.strip().lower().replace(' ', '').replace('-', '')
    
    # Vectorized pandas match: normalize the dataframe column for comparison
    match = df[df['title'].str.lower().str.replace(' ', '').str.replace('-', '') == clean_user_name]
    
    if not match.empty:
        return match.index[0]
    return -1

# def get_movie_title(i):
#     if i > len(df):
#         return ""
#     else:
#         return df.loc[i, 'title']

if st.button("Recommend Movies"):
    index = get_index_from_name(name)
    if index == -1:
        st.write("Movie not found.")
    else:
        st.write(f"Recommendations for '{name}' will be displayed here.")
        st.write(f"Movie index is { index }")
        similarity_indexes = list(enumerate(similarities[index]))
        similarity_indexes = sorted(similarity_indexes, key=lambda x: x[1], reverse=True)
        for i in range(1, 6):
            st.write(i, ":", get_name_by_index(similarity_indexes[i][0]))