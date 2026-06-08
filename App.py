import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

st.markdown("""
<style>

.stApp {
    background-color: #141414;
    color: white;
}

h1 {
    color: #E50914;
    text-align: center;
}

.stButton>button {
    background-color: #E50914;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px 20px;
}

.stButton>button:hover {
    background-color: #B20710;
}

</style>
""", unsafe_allow_html=True)

# Load dataset
df = pd.read_csv("movie_dataset (3).csv")

features = ['keywords', 'cast', 'genres', 'director']

for feature in features:
    df[feature] = df[feature].fillna('')

# Combine features
df["combined_features"] = (
    df["keywords"] + " " +
    df["cast"] + " " +
    df["genres"] + " " +
    df["director"]
)

# Vectorization
cv = CountVectorizer()
count_matrix = cv.fit_transform(df["combined_features"])

# Similarity matrix
cosine_sim = cosine_similarity(count_matrix)

# Recommendation function
def recommend(movie_name):

    movie_name = movie_name.lower()

    df["title_lower"] = df["title"].str.lower()

    if movie_name not in df["title_lower"].values:
        return []

    movie_index = df[df["title_lower"] == movie_name].index[0]

    similar_movies = list(enumerate(cosine_sim[movie_index]))

    sorted_movies = sorted(
        similar_movies,
        key=lambda x: x[1],
        reverse=True
    )[1:11]

    recommendations = []

    for movie in sorted_movies:
        recommendations.append(df.iloc[movie[0]]["title"])

    return recommendations

# Streamlit UI
st.title("🎬 Movie Recommendation System")

movie_list = sorted(df["title"].unique())

selected_movie = st.text_input(
    "Enter Movie Name"
)

if st.button("Recommend"):

    if selected_movie.strip() == "":
        st.warning("Please enter a movie name.")
    else:

        recommendations = recommend(selected_movie)

        if len(recommendations) == 0:
            st.error("Movie not found in dataset.")
        else:

            st.subheader("Recommended Movies")

            for movie in recommendations:
                st.write("🎬", movie)
