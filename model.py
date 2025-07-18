from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.chat_models import ChatOllama
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
import os
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()
# groq_api_key=os.getenv("GROQ_API_KEY")



def load_qa_chain(model,db_dir="vectorstore/", use_rag=True):
    groq_api_key=os.getenv("GROQ_API_KEY")
    # Load embeddings and vector store
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = FAISS.load_local(db_dir, embeddings, allow_dangerous_deserialization=True)

    if model in ["gemma2-9b-it", "llama-3.1-8b-instant"]:
        llm=ChatGroq(model=model, api_key=groq_api_key)
    elif model in ['D','E','F']:
        llm=ChatOllama(model=model)
    else: llm=ChatGroq(model=model, api_key=groq_api_key)

    vector_store_exists = os.path.exists(db_dir) and len(os.listdir(db_dir)) > 0

    if use_rag and vector_store_exists:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectordb = FAISS.load_local(db_dir, embeddings, allow_dangerous_deserialization=True)
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectordb.as_retriever(),
            chain_type="stuff",
            return_source_documents=True # Good for debugging or showing sources
        )
    else:
        # Fallback to a simple LLM chain for general questions
        prompt_template = PromptTemplate.from_template("{question}")
        chain = LLMChain(llm=llm, prompt=prompt_template)
    
    return chain


