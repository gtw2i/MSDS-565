import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re
from collections import Counter

# ============================================================
# Page setup
# ============================================================

st.set_page_config(
    page_title="Term Frequency Explorer",
    layout="wide"
)

st.title("Term Frequency Explorer")
st.caption("Paste text or upload a .txt file to visualize the most frequent terms.")

# ============================================================
# Input controls
# ============================================================

st.sidebar.header("Input Options")

input_mode = st.sidebar.radio(
    "Choose input method",
    ["Paste text", "Upload .txt file"]
)

st.sidebar.header("Analysis Options")

top_n = st.sidebar.slider(
    "Number of terms to show",
    min_value=5,
    max_value=50,
    value=20
)

remove_stopwords = st.sidebar.checkbox(
    "Remove common stopwords",
    value=True
)

# ============================================================
# Text input
# ============================================================

text = ""

if input_mode == "Paste text":
    text = st.text_area(
        "Paste your text here",
        height=150,
        placeholder="Paste text here..."
    )

else:
    uploaded_file = st.file_uploader(
        "Upload a .txt file",
        type=["txt"]
    )

    if uploaded_file is not None:
        text = uploaded_file.read().decode("utf-8", errors="ignore")

# ============================================================
# Stopwords
# ============================================================

stopwords = {
    "the", "and", "for", "that", "this", "with", "you", "your",
    "are", "was", "were", "but", "not", "have", "has", "had",
    "they", "them", "their", "from", "will", "would", "there",
    "what", "when", "where", "which", "who", "how", "why",
    "into", "than", "then", "also", "can", "could", "should",
    "about", "after", "before", "over", "under", "out", "our",
    "his", "her", "she", "him", "its", "all", "any", "one",
    "two", "too", "very", "just", "like", "more", "some",
    "such", "only", "own", "same"
}

# ============================================================
# Term frequency function
# ============================================================

def compute_term_frequencies(text, remove_stopwords):
    text = text.lower()

    words = re.findall(r"\b[a-zA-Z0-9']+\b", text)

    if remove_stopwords:
        words = [
            word for word in words
            if word not in stopwords
        ]

    counts = Counter(words)

    df = pd.DataFrame(
        counts.items(),
        columns=["term", "frequency"]
    )

    df = df.sort_values(
        "frequency",
        ascending=False
    ).reset_index(drop=True)

    return df

# ============================================================
# Main app logic
# ============================================================

if not text.strip():
    st.info("Paste text or upload a .txt file to begin.")
    st.stop()

freq_df = compute_term_frequencies(
    text=text,
    remove_stopwords=remove_stopwords
)

if freq_df.empty:
    st.warning("No terms found with the current settings.")
    st.stop()

top_terms = freq_df.head(top_n)

# ============================================================
# Summary metrics
# ============================================================

total_words = int(freq_df["frequency"].sum())
unique_terms = len(freq_df)
most_common_term = top_terms.iloc[0]["term"]

col1, col2, col3 = st.columns(3)

col1.metric("Total Counted Words", f"{total_words:,}")
col2.metric("Unique Terms", f"{unique_terms:,}")
col3.metric("Most Common Term", most_common_term)

st.divider()

# ============================================================
# Results
# ============================================================

tab1, tab2 = st.tabs(["Plot", "Table"])

with tab1:
    st.subheader(f"Top {top_n} Terms")

    plt.style.use("dark_background")

    plot_df = top_terms.sort_values(
        "frequency",
        ascending=True
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.barh(
        plot_df["term"],
        plot_df["frequency"]
    )

    ax.set_xlabel("Frequency")
    ax.set_ylabel("Term")
    ax.set_title(f"Top {top_n} Term Frequencies")

    ax.grid(axis="x", alpha=0.3)

    st.pyplot(fig)

with tab2:
    st.subheader("Term Frequency Table")

    st.dataframe(
        freq_df,
        use_container_width=False
    )

    st.download_button(
        label="Download frequencies as CSV",
        data=freq_df.to_csv(index=False).encode("utf-8"),
        file_name="term_frequencies.csv",
        mime="text/csv"
    )