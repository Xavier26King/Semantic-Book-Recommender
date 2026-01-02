# 📚 Semantic Book Recommender

An AI-powered book recommendation dashboard that uses semantic search to find books based on natural language descriptions.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Gradio](https://img.shields.io/badge/Gradio-4.0+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## ✨ Features

- **Semantic Search**: Describe the type of book you want in natural language
- **Category Filtering**: Filter results by book categories
- **Tone/Mood Selection**: Find books matching specific emotional tones (Happy, Sad, Suspenseful, etc.)
- **Beautiful UI**: Modern, responsive Gradio dashboard
- **Local Embeddings**: Uses HuggingFace embeddings (no API keys required for search)

## 🛠️ Tech Stack

- **Frontend**: Gradio
- **Vector Store**: FAISS
- **Embeddings**: HuggingFace Sentence Transformers
- **Framework**: LangChain

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/semantic-book-recommender.git
   cd semantic-book-recommender
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys if needed
   ```

## 🚀 Usage

Run the dashboard:
```bash
python gradio-dashboard.py
```

Open your browser and navigate to http://127.0.0.1:7860

## 📁 Project Structure

```
├── gradio-dashboard.py     # Main application
├── styles.css              # Custom CSS styling
├── books_with_emotions.csv # Book dataset with emotions
├── tagged_description.txt  # Book descriptions for embeddings
├── requirements.txt        # Python dependencies
├── .env.example           # Example environment variables
├── .gitignore             # Git ignore rules
└── README.md              # This file
```

## 🎯 How It Works

1. Book descriptions are converted into vector embeddings using HuggingFace models
2. User queries are also converted to embeddings
3. FAISS performs similarity search to find matching books
4. Results are filtered by category and emotional tone
5. Book covers and details are displayed in a gallery format

## 📝 License

This project is licensed under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
