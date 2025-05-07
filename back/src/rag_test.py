import time
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from dotenv import load_dotenv, find_dotenv

load_dotenv(dotenv_path=find_dotenv('.env.development'))

start_time = time.time()
loader = DirectoryLoader(path="back/docs")
docs = loader.load()
print(f"all docs loaded in {time.time() - start_time:.2f} seconds")

start_time = time.time()
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=100, chunk_overlap=50
)
doc_splits = text_splitter.split_documents(docs)
print(f"all docs split in {time.time() - start_time:.2f} seconds")

start_time = time.time()
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=OpenAIEmbeddings(),
)
print(f"vectorstore created in {time.time() - start_time:.2f} seconds")

start_time = time.time()
retriever = vectorstore.as_retriever()
query = "Cual es la tactica Dividir y paralelizar"
results = retriever.invoke(query)
print(f"retriever tested in {time.time() - start_time:.2f} seconds")

for result in results:
    print("=============")
    print(result)