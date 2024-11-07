
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


LANGCHAIN_TRACING_V2="true"
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY=""
LANGCHAIN_PROJECT=""
OPENAI_API_KEY=""
GOOGLE_API_KEY=""


os.environ.setdefault("OPENAI_API_KEY", OPENAI_API_KEY)

print("Creating text splitter")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
with open("source.txt", "r") as src:
    data = src.read()
all_splits = text_splitter.split_text(data)
print(all_splits[:3])


print("Creating vector db")
vectorstore = Chroma.from_texts(
    texts=all_splits,
    embedding=OpenAIEmbeddings(),
    persist_directory="vector_store")
