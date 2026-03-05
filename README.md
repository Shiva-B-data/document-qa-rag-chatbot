# Document Q&A Chatbot — RAG Pipeline

A production-style document question-answering chatbot built with LangChain, 
ChromaDB, and Google Gemini. Upload any PDF and ask questions about it in 
natural language.

## How It Works

1. Upload a PDF document
2. The app splits it into chunks and generates embeddings using Google Gemini
3. Embeddings are stored in ChromaDB (vector database)
4. When you ask a question, the most relevant chunks are retrieved
5. Google Gemini generates an answer based on the retrieved context

## Tech Stack

- **LangChain** — RAG pipeline orchestration
- **ChromaDB** — Vector database for storing and retrieving embeddings
- **Google Gemini** — LLM for embeddings and answer generation
- **Streamlit** — Frontend UI
- **Python** — Core language

## Setup

1. Clone the repository
   git clone https://github.com/Shiva-B-data/document-qa-rag-chatbot.git
   cd document-qa-rag-chatbot

2. Create a virtual environment
   python -m venv venv
   venv\Scripts\activate

3. Install dependencies
   pip install -r requirements.txt

4. Add your Google Gemini API key
   Create a .env file and add:
   GOOGLE_API_KEY=your-key-here

5. Run the app
   streamlit run app.py

## Author

Shiva Boddu — Machine Learning Engineer
linkedin.com/in/shiva-b-012a372b9

