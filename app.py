import streamlit as st
import os
import tempfile
from rag_pipeline import (
    load_and_split_pdf,
    create_vector_store,
    build_qa_chain,
    answer_question
)

st.set_page_config(
    page_title="Document Q&A Chatbot",
    page_icon="📄",
    layout="centered"
)

st.title("📄 Document Q&A Chatbot")
st.write("Upload a PDF and ask questions about it — powered by RAG, LangChain and Google Gemini")

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file is not None:
    if st.button("Process Document"):
        with st.spinner("Reading and indexing your document..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            chunks = load_and_split_pdf(tmp_path)
            vector_store = create_vector_store(chunks)
            st.session_state.qa_chain = build_qa_chain(vector_store)
            os.unlink(tmp_path)

        st.success(f"Done! Document split into {len(chunks)} chunks and indexed.")

if st.session_state.qa_chain:
    st.divider()
    st.subheader("Ask a question about your document")

    question = st.text_input(
        "Your question:",
        placeholder="e.g. What is the main topic of this document?"
    )

    if st.button("Ask") and question:
        with st.spinner("Thinking..."):
            answer, _ = answer_question(
                st.session_state.qa_chain, question
            )
        st.session_state.chat_history.append({
            "question": question,
            "answer": answer
        })

    if st.session_state.chat_history:
        st.divider()
        st.subheader("Conversation")
        for chat in reversed(st.session_state.chat_history):
            st.markdown(f"**You:** {chat['question']}")
            st.markdown(f"**Bot:** {chat['answer']}")
            st.write("")
else:
    st.info("Upload and process a PDF to get started.")

st.divider()
st.caption("Built with LangChain, ChromaDB, Google Gemini, and Streamlit")
