import streamlit as st
from ingest import create_vector_store
from model import load_qa_chain
import os
from langchain.chains import RetrievalQA

st.title("Multi-Model LLM Prompt Service")

uploaded_file = st.file_uploader("📤 Upload a PDF file", type="pdf")

# State variable to track if PDF has been processed
if 'pdf_processed' not in st.session_state:
    st.session_state.pdf_processed = False
# Define a default database directory
DB_DIR = "vectorstore/"

if uploaded_file and not st.session_state.pdf_processed:
    file_path = os.path.join("docs", uploaded_file.name)
    os.makedirs("docs", exist_ok=True)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("🔍 Processing and indexing your PDF..."):
        create_vector_store(file_path, db_dir=DB_DIR) # Pass the specific DB_DIR
    st.success("✅ PDF indexed successfully!")
    st.session_state.pdf_processed = True
    st.rerun() # Rerun to update the state immediately
elif st.session_state.pdf_processed and uploaded_file is not None:
    st.info("PDF already processed. You can now ask questions based on the document or general questions.")
elif not st.session_state.pdf_processed and uploaded_file is None:
    st.info("No PDF uploaded. You can still ask general questions to the LLM.")


st.sidebar.header("Model Configuration")
llm_model_name = st.sidebar.selectbox("Select Open Source Model", ["gemma2-9b-it", "llama-3.1-8b-instant"])

st.markdown(f"Hello! I'm {llm_model_name}. Ask me anything")
user_input = st.text_input("You:")

if user_input:
    with st.spinner(f"Thinking with {llm_model_name}..."):
        # Determine whether to use RAG based on pdf_processed state
        use_rag_mode = st.session_state.pdf_processed and os.path.exists(DB_DIR) and len(os.listdir(DB_DIR)) > 0
        
        chain = load_qa_chain(llm_model_name, db_dir=DB_DIR, use_rag=use_rag_mode)
        
        # Determine the input key for the chain based on its type
        if isinstance(chain, RetrievalQA):
            result = chain.invoke({'query': user_input}) # RetrievalQA often expects 'query' or 'question'. 'query' is common.
            st.subheader("Response (From Document)")
            st.write(result["result"])
        else: # It's an LLMChain
            result = chain.invoke({'question': user_input}) 
            st.subheader("Response")
            # LLMChain result is typically just the output string or in a 'text' key
            st.write(result.get("text", result))
else:
    st.info("Please enter a question.")



