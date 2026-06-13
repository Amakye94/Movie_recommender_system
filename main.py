import pandas as pd
import numpy as np
import os
import joblib

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
# CREATE AVERAGE RATINGS DICTIONARY
# =====================================

movie_avg_ratings = (
    ratings.groupby("movieId")["rating"]
    .mean()
    .to_dict()
)

del ratings

print("Average ratings dictionary created")

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
# LOAD PRE-TRAINED SVD MODEL
# =====================================

print("Loading SVD model...")

model = joblib.load(
    "svd_model.pkl"
)

print("SVD model loaded")

# =====================================
# LOAD PRE-COMPUTED EMBEDDINGS
# =====================================

print("Loading embeddings...")

embeddings = np.load(
    "embeddings.npy"
)

print("Embeddings loaded")

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

        rating_score = movie_avg_ratings.get(
            movie_id,
            3.0
        )

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

ui.colors(primary="#E50914")

with ui.column().classes(
    "w-full items-center"
):

    ui.label(
        "🎬 Hybrid Movie Recommender System"
    ).classes(
        "text-h3 font-bold"
    )

    ui.label(
        "MovieLens 1M + BERT + Hybrid Recommendation"
    ).classes(
        "text-subtitle1"
    )

movie_dropdown = ui.select(
    options=sorted(
        movies["title"].tolist()
    ),
    label="Select a Movie"
).classes(
    "w-96"
)

results_container = ui.column().classes(
    "w-full"
)

def show_recommendations():

    results_container.clear()

    if not movie_dropdown.value:

        ui.notify(
            "Please select a movie",
            color="negative"
        )

        return

    recommendations = recommend_movies(
        movie_dropdown.value
    )

    with results_container:

        ui.separator()

        ui.label(
            f"Recommendations for {movie_dropdown.value}"
        ).classes(
            "text-h5"
        )

        for rank, (movie, score) in enumerate(
            recommendations,
            start=1
        ):

            with ui.card().classes(
                "w-full"
            ):

                ui.label(
                    f"#{rank} 🎬 {movie}"
                ).classes(
                    "text-h6 font-bold"
                )

                ui.label(
                    f"Hybrid Score: {score:.3f}"
                )

ui.button(
    "🚀 Get Recommendations",
    on_click=show_recommendations
)

ui.run(
    host="0.0.0.0",
    port=int(os.environ.get("PORT", 8080)),
    title="Hybrid Movie Recommender",
    reload=False
)

