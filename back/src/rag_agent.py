from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from dotenv import load_dotenv, find_dotenv

load_dotenv(dotenv_path=find_dotenv('.env.development'))

# Load documents from the local 'docs' folder
loader = DirectoryLoader(path="docs")
docs = loader.load()

print("all docs loaded")

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=100, chunk_overlap=50
)
doc_splits = text_splitter.split_documents(docs)

print("all docs split")

vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=OpenAIEmbeddings(),
)

print("vectorstore created")

retriever = vectorstore.as_retriever()

# Test the retriever
query = "What is encapsulation"
results = retriever.invoke(query)
for result in results:
    print("=============")
    print(result)