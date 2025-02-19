from langchain_community.document_loaders import GoogleCloudStorageLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from google.cloud import storage

bucket_name = "add-knowledge-db"
prefix = "Training Docs/"  

storage_client = storage.Client()

loader = GoogleCloudStorageLoader(client=storage_client, 
                                  bucket_name=bucket_name, 
                                  prefix=prefix)
docs = loader.load()


docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=100, chunk_overlap=50
)
doc_splits = text_splitter.split_documents(docs_list)

vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=OpenAIEmbeddings(),
)

retriever = vectorstore.as_retriever()


