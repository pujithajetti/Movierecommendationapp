# 🎬 Movie Recommendation Web App

An interactive web application built with **Streamlit** that recommends movies based on content similarity using the TMDB dataset and cosine similarity. Pre-trained models are used to generate personalized movie suggestions.

---

## ℹ️ Overview

- **Dataset**: TMDB 5000 dataset (`tmdb_5000_movies.csv`, `tmdb_5000_credits.csv`)
- **Technique**: Content-based filtering using TF‑IDF vectorization and cosine similarity
- **UI**: Built with Streamlit — users can search for a movie and receive similar movie recommendations

---

## 🧰 Features

- **Search by Title**: Input a movie name to get relevant recommendations.
- **Top-N Recommendations**: Displays the top 5 most similar movies.
- **Movie Metadata**: Can include static posters and release years.

---

## 📦 Repository Structure

```
Movierecommendationapp/
├── app.py                     # Streamlit web app interface
├── model.pkl                  # Serialized similarity model
├── tmdb_5000_movies.csv       # Movie metadata
├── tmdb_5000_credits.csv      # Cast and crew data
├── recommendation.ipynb       # Jupyter notebook for training and analysis
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

---

## ⚙️ Setup & Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/pujithajetti/Movierecommendationapp.git
   cd Movierecommendationapp
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

The application will open in your default web browser at `http://localhost:8501`.

---

## 🧠 How It Works

1. Loads and preprocesses the movie data from TMDB dataset.
2. Combines features like genres, overview, and cast into a text field.
3. Vectorizes the text using TF‑IDF.
4. Calculates pairwise cosine similarity between all movies.
5. When a movie is selected, retrieves and displays the most similar movies.

---

## ✅ Future Enhancements

- Add support for TMDB API integration for dynamic posters
- Add filters such as release year or rating
- Improve UI with more detailed information per movie
- Include collaborative filtering or hybrid recommendation system

---
