import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

#  Load Model #
@st.cache_data
def load_model():
    try:
        with open('movie_recommendation_model.sav', 'rb') as f:
            model_data = pickle.load(f)

        # Unwrap if needed (common mistake: saved as [[df]] or [df])
        if isinstance(model_data, list) and isinstance(model_data[0], dict):
            model_data = model_data[0]
        elif isinstance(model_data, pd.DataFrame):
            # Should not be a DataFrame directly — raise error
            st.error("❌ Loaded object appears to be a raw DataFrame, not model data.")
            return None

        return model_data
    except FileNotFoundError:
        st.error("❌ Model file not found. Please check the path and filename.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

# ----------------- Recommendation Logic ----------------- #
def compute_similarity(input_idx, reduced_matrix, top_k=10):
    similarity_vector = cosine_similarity(
        [reduced_matrix[input_idx]], reduced_matrix
    )[0]
    similar_indices = similarity_vector.argsort()[::-1][1:top_k+1]
    return similar_indices

def recommend_movies(model_data, input_title, top_k=10):
    reduced_matrix = model_data['reduced_matrix']
    title_to_index = model_data['title_to_index']
    titles_df = model_data['titles']
    titles = titles_df['title'].tolist()

    matches = [title for title in titles if input_title.lower() in title.lower()]
    if not matches:
        return None, pd.DataFrame()

    selected_title = matches[0]
    movie_idx = title_to_index[selected_title]
    similar_indices = compute_similarity(movie_idx, reduced_matrix, top_k)

    recommendations = titles_df.iloc[similar_indices]
    return selected_title, recommendations

# Streamlit UI #
st.set_page_config(page_title="🎬 Movie Recommender", layout="centered")
st.title("🎥 Movie Recommendation System")
st.markdown("Type a movie name you like, and get recommendations based on content similarity!")

model_data = load_model()

if model_data:
    # User input
    user_input = st.text_input("Enter a movie title:", "")

    if user_input:
        selected_title, recommendations = recommend_movies(model_data, user_input, top_k=10)

        if recommendations is None or recommendations.empty:
            st.error("No matching movie found. Try a different title.")
        else:
            st.success(f"Top 10 movies similar to **{selected_title}**:")
            for i, row in recommendations.iterrows():
                st.markdown(f"**{i+1}. {row['title']}**")
