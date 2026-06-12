
import pandas as pd
import numpy as np
import os

from surprise import Dataset
from surprise import Reader
from surprise import SVD

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nicegui import ui

# =====================================
# LOAD MOVIELENS 1M DATA
# =====================================

movies = pd.read_csv(
    "movies.dat",
    sep="::",
    engine="python",
    encoding="latin-1",
    names=[
        "movieId",
        "title",
        "genres"
    ]
)

ratings = pd.read_csv(
    "ratings.dat",
    sep="::",
    engine="python",
    names=[
        "userId",
        "movieId",
        "rating",
        "timestamp"
    ]
)

print("Movies:", len(movies))
print("Ratings:", len(ratings))

# =====================================
# CREATE CONTENT COLUMN
# =====================================

movies["content"] = (
    movies["title"].fillna("")
    + " "
    + movies["genres"].fillna("")
)

print("Content column created")

# =====================================
# TRAIN SVD MODEL
# =====================================

reader = Reader(
    rating_scale=(1, 5)
)

data = Dataset.load_from_df(
    ratings[
        ["userId", "movieId", "rating"]
    ],
    reader
)

trainset = data.build_full_trainset()

model = SVD()

print("Training SVD...")

model.fit(trainset)

print("SVD Training Complete!")

# =====================================
# LOAD BERT MODEL
# =====================================

print("Loading BERT model...")

bert_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# =====================================
# LOAD OR CREATE EMBEDDINGS
# =====================================

if os.path.exists("embeddings.npy"):

    embeddings = np.load(
        "embeddings.npy"
    )

    print(
        "Embeddings loaded from cache"
    )

else:

    print(
        "Generating embeddings..."
    )

    embeddings = bert_model.encode(
        movies["content"].tolist(),
        show_progress_bar=True
    )

    np.save(
        "embeddings.npy",
        embeddings
    )

    print(
        "Embeddings generated and saved"
    )

# =====================================
# HYBRID RECOMMENDATION FUNCTION
# =====================================

def recommend_movies(movie_title):

    movie_row = movies[
        movies["title"] == movie_title
    ]

    if movie_row.empty:
        return []

    movie_index = movie_row.index[0]

    movie_embedding = embeddings[
        movie_index
    ].reshape(1, -1)

    similarities = cosine_similarity(
        movie_embedding,
        embeddings
    )[0]

    recommendations = []

    for idx, row in movies.iterrows():

        if idx == movie_index:
            continue

        movie_id = row["movieId"]

        avg_rating = ratings[
            ratings["movieId"] == movie_id
        ]["rating"].mean()

        if pd.isna(avg_rating):
            avg_rating = 3.0

        rating_score = avg_rating

        bert_score = similarities[idx]

        hybrid_score = (
            0.7 * (rating_score / 5)
            + 0.3 * bert_score
        )

        recommendations.append(
            (
                row["title"],
                hybrid_score
            )
        )

    recommendations.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return recommendations[:10]

# =====================================
# NICEGUI INTERFACE
# =====================================

ui.colors(primary='#E50914')

with ui.column().classes(
    'w-full items-center'
):

    ui.label(
        '🎬 Hybrid Movie Recommender System'
    ).classes(
        'text-h3 font-bold'
    )

    ui.label(
        'MovieLens 1M + BERT + Hybrid Recommendation'
    ).classes(
        'text-subtitle1'
    )

movie_dropdown = ui.select(
    options=sorted(
        movies["title"].tolist()
    ),
    label='Select a Movie'
).classes(
    'w-96'
)

results_container = ui.column().classes(
    'w-full'
)

def show_recommendations():

    results_container.clear()

    if not movie_dropdown.value:

        ui.notify(
            'Please select a movie',
            color='negative'
        )

        return

    recommendations = recommend_movies(
        movie_dropdown.value
    )

    with results_container:

        ui.separator()

        ui.label(
            f'Recommendations for {movie_dropdown.value}'
        ).classes(
            'text-h5'
        )

        for rank, (movie, score) in enumerate(
            recommendations,
            start=1
        ):

            with ui.card().classes(
                'w-full'
            ):

                ui.label(
                    f'#{rank} 🎬 {movie}'
                ).classes(
                    'text-h6 font-bold'
                )

                ui.label(
                    f'Hybrid Score: {score:.3f}'
                )

ui.button(
    '🚀 Get Recommendations',
    on_click=show_recommendations
)

ui.run(
    title='Hybrid Movie Recommender',
    reload=False,
    port=8080
)