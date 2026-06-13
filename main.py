import pandas as pd
import numpy as np
import os
import requests

from sklearn.metrics.pairwise import cosine_similarity
from nicegui import ui



# =====================================
# TMDB CONFIG
# =====================================

TMDB_API_KEY = os.environ.get("TMDB_API_KEY")

if TMDB_API_KEY:
    print("TMDB Key Found")
else:
    print("TMDB Key NOT Found")


def get_movie_info(movie_title):

    try:

        url = (
            "https://api.themoviedb.org/3/search/movie"
            f"?api_key={TMDB_API_KEY}"
            f"&query={movie_title}"
        )

        response = requests.get(
            url,
            timeout=10
        )

        if response.status_code != 200:

            print(
                "TMDB API Error:",
                response.status_code
            )

            return None

        data = response.json()

        if len(data["results"]) == 0:
            return None

        movie = data["results"][0]

        poster = None

        if movie.get("poster_path"):

            poster = (
                "https://image.tmdb.org/t/p/w500"
                + movie["poster_path"]
            )

        return {
            "overview": movie.get(
                "overview",
                "No description available."
            ),

            "release_date": movie.get(
                "release_date",
                "Unknown"
            ),

            "rating": movie.get(
                "vote_average",
                "N/A"
            ),

            "poster": poster
        }

    except Exception as e:

        print(
            "TMDB Error:",
            e
        )

        return None



# =====================================
# LOAD MOVIELENS DATA
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
# CREATE AVERAGE RATINGS
# =====================================

movie_avg_ratings = (
    ratings.groupby("movieId")["rating"]
    .mean()
    .to_dict()
)

del ratings

print("Average ratings dictionary created")

# =====================================
# CONTENT COLUMN
# =====================================

movies["content"] = (
    movies["title"].fillna("")
    + " "
    + movies["genres"].fillna("")
)

print("Content column created")

# =====================================
# LOAD EMBEDDINGS
# =====================================

print("Loading embeddings...")

embeddings = np.load(
    "embeddings.npy"
)

print("Embeddings loaded")

# =====================================
# RECOMMENDATION FUNCTION
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

        similarity_score = similarities[idx]

        hybrid_score = (
            0.7 * (rating_score / 5)
            + 0.3 * similarity_score
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
# UI DESIGN
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
        "MovieLens + TMDB Enhanced Recommendations"
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

# =====================================
# DISPLAY RECOMMENDATIONS
# =====================================
def get_movie_trailer(movie_title):

    try:

        search_url = (
            f"https://api.themoviedb.org/3/search/movie"
            f"?api_key={TMDB_API_KEY}"
            f"&query={movie_title}"
        )

        movie_data = requests.get(
            search_url
        ).json()

        if not movie_data["results"]:
            return None

        movie_id = movie_data["results"][0]["id"]

        video_url = (
            f"https://api.themoviedb.org/3/movie/"
            f"{movie_id}/videos"
            f"?api_key={TMDB_API_KEY}"
        )

        videos = requests.get(
            video_url
        ).json()

        for video in videos["results"]:

            if (
                video["site"] == "YouTube"
                and video["type"] == "Trailer"
            ):

                return (
                    "https://www.youtube.com/watch?v="
                    + video["key"]
                )

        return None

    except Exception as e:

        print(
            "Trailer Error:",
            e
        )

        return None
    
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

                clean_title = movie.split(
                    " ("
                )[0].strip()

                movie_info = get_movie_info(
                    clean_title
                )

                if movie_info:

                    with ui.row():

                        if movie_info["poster"]:

                            ui.image(
                                movie_info["poster"]
                            ).classes(
                                "w-40"
                            )

                        with ui.column():

                            ui.label(
                                f"⭐ TMDB Rating: {movie_info['rating']}"
                            )

                            ui.label(
                                f"📅 Release Date: {movie_info['release_date']}"
                            )

                            ui.label(
                                f"📝 {movie_info['overview']}"
                            )

                ui.label(
                    f"🔥 Recommendation Score: {score:.3f}"
                )

                trailer_url = get_movie_trailer(
                    clean_title
                )

                if trailer_url:

                    ui.link(
                        "🎥 Watch Trailer",
                        trailer_url
                    ).classes(
                        "text-blue"
                    )
   
# =====================================
# BUTTON
# =====================================

ui.button(
    "🚀 Get Recommendations",
    on_click=show_recommendations
)


# =====================================
# RUN APP
# =====================================

ui.run(
    host="0.0.0.0",
    port=int(os.environ.get("PORT", 8080)),
    title="Hybrid Movie Recommender",
    reload=False
)

