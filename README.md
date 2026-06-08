# 🎬 Movie Recommendation System

A Content-Based Movie Recommendation System built using Python, Streamlit, and Machine Learning techniques. The system recommends movies based on similarities in genres, keywords, cast members, and directors.

---

## 📌 Features

* Movie recommendation based on content similarity
* Uses Cosine Similarity and Count Vectorization
* Interactive web interface with Streamlit
* Search for movies and receive similar movie suggestions
* Fast and lightweight deployment

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Streamlit

---

## 📂 Dataset

The project uses a movie dataset containing:

* Movie Title
* Genres
* Keywords
* Cast
* Director

These features are combined and transformed into numerical vectors using CountVectorizer.

---

## ⚙️ How It Works

### Step 1: Data Preprocessing

The following movie attributes are selected:

* Keywords
* Cast
* Genres
* Director

The selected features are combined into a single text feature.

### Step 2: Feature Extraction

CountVectorizer converts textual movie information into a matrix representation.

### Step 3: Similarity Calculation

Cosine Similarity is used to calculate similarity scores between movies.

### Step 4: Recommendation Generation

When a user enters or selects a movie, the system identifies the most similar movies and recommends them.

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/Amakye94/Movie_recommender_system.git
```

Move into the project directory:

```bash
cd Movie_recommender_system
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run App.py
```

---

## 📸 Application Preview

The application allows users to:

1. Search for a movie.
2. Click the recommendation button.
3. Receive a list of similar movies.

---

## 📊 Machine Learning Techniques

* Content-Based Filtering
* Count Vectorization
* Cosine Similarity

---

## 🔮 Future Improvements

* Movie poster integration
* TMDB API integration
* Movie ratings display
* User authentication
* Hybrid recommendation system
* Collaborative filtering implementation

---

## 👨‍💻 Author

**Ebenezer Asamoah Amakye**

GitHub: https://github.com/Amakye94

---

## 📜 License

This project is developed for educational and learning purposes.
