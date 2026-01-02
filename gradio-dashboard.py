import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import gradio as gr

load_dotenv()

books = pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"],
)


from langchain_core.documents import Document

with open("tagged_description.txt", "r") as f:
    lines = f.read().strip().split("\n")

documents = []
for line in lines:
    if line.strip():
        documents.append(Document(page_content=line, metadata={"source": "tagged_description.txt"}))

db_books = FAISS.from_documents(documents, HuggingFaceEmbeddings())


def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:

    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)

    return book_recs


def recommend_books(
        query: str,
        category: str,
        tone: str
):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))
    return results

categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

# Load custom CSS from external file
with open("styles.css", "r") as f:
    custom_css = f.read()

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as dashboard:
    # Header Section
    gr.Markdown(
        """
        <div style="text-align: center; padding: 2rem 0;">
            <h1 class="header-title">📚 Semantic Book Recommender</h1>
            <p class="header-subtitle">Discover your next favorite book using AI-powered semantic search</p>
        </div>
        """
    )
    
    # Search Section
    with gr.Group():
        gr.Markdown("### 🔍 What kind of book are you looking for?")
        user_query = gr.Textbox(
            label="",
            placeholder="Describe the book you're looking for... (e.g., 'A heartwarming story about friendship and adventure')",
            lines=2,
            elem_classes=["search-box"]
        )
    
    # Filters Section
    gr.Markdown('<p class="section-title">🎛️ Refine Your Search</p>')
    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            category_dropdown = gr.Dropdown(
                choices=categories,
                label="📂 Category",
                value="All",
                info="Filter by book genre"
            )
        with gr.Column(scale=1):
            tone_dropdown = gr.Dropdown(
                choices=tones,
                label="🎭 Emotional Tone",
                value="All",
                info="Filter by the mood of the book"
            )
        with gr.Column(scale=1, min_width=200):
            submit_button = gr.Button(
                "✨ Find Books",
                variant="primary",
                elem_classes=["primary-btn"],
                size="lg"
            )
    
    # Results Section
    gr.Markdown('<p class="section-title">📖 Recommended Books</p>')
    output = gr.Gallery(
        label="",
        columns=4,
        rows=2,
        height="auto",
        object_fit="contain",
        elem_classes=["gallery-container"],
        show_label=False
    )
    
    # Footer
    gr.Markdown(
        """
        <footer>
            <p>Powered by HuggingFace Embeddings & FAISS Vector Search</p>
            <p style="font-size: 0.85rem;">💡 Tip: Be descriptive in your search for better recommendations!</p>
        </footer>
        """
    )

    # Event handler
    submit_button.click(
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=output
    )
    
    # Allow Enter key to submit
    user_query.submit(
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=output
    )


if __name__ == "__main__":
    dashboard.launch()