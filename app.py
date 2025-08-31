import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# 1. Load the preprocessed dataset

@st.cache_data
def load_data():
    # Load cleaned dataset
    df = pd.read_csv("imdb_movies_storylines_cleaned.csv")
    return df

df = load_data()


# 2. Generate TF-IDF matrix and cosine similarity

@st.cache_resource
def compute_similarity(data):
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(data["Cleaned_Storyline"])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return tfidf, cosine_sim

tfidf, cosine_sim = compute_similarity(df)


# 3. Recommendation function (ensures unique movies)

def recommend_movies_from_storyline(input_storyline, df, tfidf, cosine_sim):
    # Convert input storyline into TF-IDF vector
    input_vec = tfidf.transform([input_storyline])

    # Compute similarity between input and all movies
    sim_scores = cosine_similarity(input_vec, tfidf.transform(df["Cleaned_Storyline"])).flatten()

    # Sort movies by similarity (highest first)
    top_indices = sim_scores.argsort()[::-1]

    # Create a DataFrame with similarity scores
    results = df.iloc[top_indices].copy()
    results["Similarity"] = sim_scores[top_indices]

    # Drop duplicates based on Movie Name
    results = results.drop_duplicates(subset="Movie Name")

    # Take top 5 unique movies
    recommendations = results.head(5)[["Movie Name", "Storyline"]]

    return recommendations


# 4. Streamlit UI

st.set_page_config(page_title="🎬 IMDb Movie Recommendation System", layout="wide")

st.title("🎥 IMDb Movie Recommendation System")
st.write("Enter a **movie storyline** and get **top 5 recommended movies** based on plot similarity.")

# User input for storyline
user_storyline = st.text_area("✍️ Enter a movie storyline or short description:", height=150)

# Recommend button
if st.button("🔍 Get Recommendations"):
    if user_storyline.strip() == "":
        st.warning("⚠️ Please enter a storyline to get recommendations.")
    else:
        recommendations = recommend_movies_from_storyline(user_storyline, df, tfidf, cosine_sim)

        st.subheader("📌 Top 5 Recommended Movies")
        for idx, row in recommendations.iterrows():
            st.markdown(f"### 🎬 {row['Movie Name']}")
            st.write(row["Storyline"])
            st.markdown("---")
