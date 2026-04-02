import streamlit as st
import pandas as pd
import pickle
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import gdown

# -----------------------------
# LOAD EVERYTHING SAFELY
# -----------------------------
@st.cache_resource
def load_all():

    def download(file_id, output):
        if not os.path.exists(output):
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, output, quiet=False, fuzzy=True)

    # 🔥 Download files INSIDE cache
    download("1aX1nG0veHY7ydmeBGKNTT9nXCMc41Cqf", "df_cb.pkl")
    download("1ZQtSeT4l7nFwHKcUMxdJYw207gW4LQ3x", "cv.pkl")
    download("1kWakzDE21nU-rplidm_RN4sc4nyWZ5Sd", "vectors.pkl")
    download("19DNrX3z2PBA2LPApSsCpJ3FGS53wIxF9", "indices.pkl")
    download("1XDhcEuHe-H4-D7dRND11EB0_z2RWMZaW", "co_occurrence.pkl")

    # 🔥 Load files
    df_cb = pd.read_pickle("df_cb.pkl")
    tfidf = pickle.load(open("cv.pkl", "rb"))
    tfidf_matrix = pickle.load(open("vectors.pkl", "rb"))
    indices = pickle.load(open("indices.pkl", "rb"))
    co_occurrence = pickle.load(open("co_occurrence.pkl", "rb"))

    return df_cb, tfidf, tfidf_matrix, indices, co_occurrence


# -----------------------------
# UI SETTINGS
# -----------------------------
st.set_page_config(page_title="🛍️ Recommender", layout="wide")

st.title("🛍️ Smart Product Recommender")
st.markdown("#### 🔎 Search products and get smart recommendations instantly")

# 🔥 Load AFTER UI (prevents 502 crash)
with st.spinner("Loading model... please wait ⏳"):
    df_cb, tfidf, tfidf_matrix, indices, co_occurrence = load_all()

# -----------------------------
# ENSURE REAL RATINGS
# -----------------------------
if 'rating' not in df_cb.columns:
    st.error("❌ Rating column missing! Please compute from reviews dataset.")
    st.stop()

df_cb['rating'] = df_cb['rating'].fillna(3.5)

if 'num_reviews' in df_cb.columns:
    df_cb['num_reviews'] = df_cb['num_reviews'].fillna(0)

# -----------------------------
# ⭐ STAR FUNCTION
# -----------------------------
def render_stars(rating):
    full = int(rating)
    half = 1 if rating - full >= 0.5 else 0
    empty = 5 - full - half
    return "⭐" * full + "☆" * (half + empty)

# -----------------------------
# PREPROCESS
# -----------------------------
def preprocess(text):
    text = str(text).lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [word for word in words if len(word) > 2]
    return " ".join(words)

# -----------------------------
# SEARCH
# -----------------------------
def search_products(query, top_n=10):
    query = preprocess(query)
    query_vec = tfidf.transform([query])
    sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = sim_scores.argsort()[-top_n:][::-1]
    return df_cb.iloc[top_indices]

# -----------------------------
# RECOMMEND (CONTENT)
# -----------------------------
def recommend_similar_products(asin, top_n=5):
    if asin not in indices.index:
        return pd.DataFrame()

    idx = indices[asin]
    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    sim_indices = sim_scores.argsort()[-top_n-1:-1][::-1]
    return df_cb.iloc[sim_indices]

# -----------------------------
# RECOMMEND (ASSOCIATION)
# -----------------------------
def recommend_association(asin, top_n=5):
    if asin not in co_occurrence:
        return pd.DataFrame()

    related_items = co_occurrence[asin]
    sorted_items = sorted(related_items.items(), key=lambda x: x[1], reverse=True)
    top_items = [item for item, _ in sorted_items[:top_n]]

    return df_cb[df_cb['asin'].isin(top_items)][['asin', 'title', 'image', 'rating', 'num_reviews']]

# -----------------------------
# SESSION STATE
# -----------------------------
if 'selected_asin' not in st.session_state:
    st.session_state.selected_asin = None

# -----------------------------
# SEARCH BAR
# -----------------------------
query = st.text_input("Search for products (e.g. sneakers, shirts, shoes)")

# -----------------------------
# SEARCH RESULTS
# -----------------------------
if query:
    results = search_products(query)
    st.markdown("### 🔍 Search Results")

    cols = st.columns(5)

    for i, (_, row) in enumerate(results.iterrows()):
        with cols[i % 5]:

            title = row['title'][:50] + "..."
            rating = row['rating']
            num_reviews = int(row.get('num_reviews', 0))
            stars = render_stars(rating)
            rating = f"{row['rating']:.1f}"

            st.image(row['image'], width=150)

            st.markdown(f"""
            <p style="font-size:14px; height:40px; overflow:hidden;">
                {title}
            </p>
            <p style="color:#f5a623; font-weight:bold;">
                {stars} {rating} ({num_reviews})
            </p>
            """, unsafe_allow_html=True)

            if st.button("View Similar", key=f"btn_search_{row['asin']}"):
                st.session_state.selected_asin = row['asin']

# -----------------------------
# RECOMMENDATIONS
# -----------------------------
if st.session_state.selected_asin:

    st.markdown("---")

    # 🎯 Similar Products
    st.markdown("### 🎯 Recommended Products")

    recs = recommend_similar_products(st.session_state.selected_asin)

    if not recs.empty:
        rec_cols = st.columns(5)

        for j, (_, rec) in enumerate(recs.iterrows()):
            with rec_cols[j % 5]:

                title = rec['title'][:50] + "..."
                rating = rec['rating']
                num_reviews = int(rec.get('num_reviews', 0))
                stars = render_stars(rating)
                rating = f"{rec['rating']:.1f}"

                st.image(rec['image'], width=150)

                st.markdown(f"""
                <p style="font-size:14px; height:40px; overflow:hidden;">
                    {title}
                </p>
                <p style="color:#f5a623; font-weight:bold;">
                    {stars} {rating} ({num_reviews})
                </p>
                """, unsafe_allow_html=True)

                if st.button("View Similar", key=f"btn_rec_{rec['asin']}"):
                    st.session_state.selected_asin = rec['asin']

    else:
        st.info("No similar products found.")

    # 🛒 Frequently Bought Together
    st.markdown("### 🛒 Frequently Bought Together")

    assoc = recommend_association(st.session_state.selected_asin)

    if not assoc.empty:
        assoc_cols = st.columns(5)

        for k, (_, item) in enumerate(assoc.iterrows()):
            with assoc_cols[k % 5]:

                title = item['title'][:50] + "..."
                rating = item['rating']
                num_reviews = int(item.get('num_reviews', 0))
                stars = render_stars(rating)
                rating = f"{item['rating']:.1f}"

                st.image(item['image'], width=150)

                st.markdown(f"""
                <p style="font-size:14px; height:40px; overflow:hidden;">
                    {title}
                </p>
                <p style="color:#f5a623; font-weight:bold;">
                    {stars} {rating} ({num_reviews})
                </p>
                """, unsafe_allow_html=True)

                if st.button("View Similar", key=f"btn_assoc_{item['asin']}"):
                    st.session_state.selected_asin = item['asin']

    else:
        st.info("No association data found for this product.")