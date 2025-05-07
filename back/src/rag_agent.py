from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(dotenv_path=find_dotenv('.env.development'))

CHROMA_DB_DIR = "./chroma_db"
DOCS_DIR = "../back/docs" # para ejecucion del archivo quitar los dos puntos
COLLECTION_NAME = "rag-chroma"

def create_or_load_vectorstore():
    """
    Creates a new Chroma database if it doesn't exist,
    or loads an existing one if it does.
    """
    # Check if the database already exists
    embedding = OpenAIEmbeddings()
    
    if os.path.exists(CHROMA_DB_DIR) and os.listdir(CHROMA_DB_DIR):
        print(f"Loading existing Chroma DB from {CHROMA_DB_DIR}")
        # Load the existing database
        vectorstore = Chroma(
            persist_directory=CHROMA_DB_DIR,
            embedding_function=embedding,
            collection_name=COLLECTION_NAME
        )
    else:
        print(f"Creating new Chroma DB at {CHROMA_DB_DIR}")
        # Create the directory if it doesn't exist
        os.makedirs(CHROMA_DB_DIR, exist_ok=True)
        
        # Load documents
        loader = DirectoryLoader(path=DOCS_DIR)
        docs = loader.load()
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=100, chunk_overlap=50
        )
        doc_splits = text_splitter.split_documents(docs)
        
        # Create and persist the vectorstore
        vectorstore = Chroma.from_documents(
            documents=doc_splits,
            collection_name=COLLECTION_NAME,
            embedding=embedding,
            persist_directory=CHROMA_DB_DIR
        )
        # Make sure to persist the database
        vectorstore.persist()
    
    return vectorstore

def get_retriever():
    """
    Returns a retriever from the vector store.
    This is the function you should call from your FastAPI endpoint.
    """
    vectorstore = create_or_load_vectorstore()
    return vectorstore.as_retriever()

def rebuild_vectorstore():
    """
    Deletes the existing database and creates a new one.
    Only use this when you want to rebuild the database (e.g., after updating documents).
    """
    import shutil
    if os.path.exists(CHROMA_DB_DIR):
        print(f"Removing existing Chroma DB at {CHROMA_DB_DIR}")
        shutil.rmtree(CHROMA_DB_DIR)
    
    return create_or_load_vectorstore()

"""
if __name__ == "__main__":
    vectorstore = create_or_load_vectorstore()
    print(f"Vector store is ready with {vectorstore._collection.count()} documents")

"""