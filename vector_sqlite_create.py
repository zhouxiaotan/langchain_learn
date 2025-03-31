from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_community.vectorstores import SQLiteVec
from langchain_text_splitters import CharacterTextSplitter

# load the document and split it into chunks
loader = TextLoader("./example_data/state_of_the_union.txt")
documents = loader.load()

# split it into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
texts = [doc.page_content for doc in docs]


# create the open-source embedding function
from langchain_ollama.embeddings import OllamaEmbeddings

embedding = OllamaEmbeddings(base_url="http://localhost:11434",model="nomic-embed-text:latest")


# load it in sqlite-vss in a table named state_union.
# the db_file parameter is the name of the file you want
# as your sqlite database.
db = SQLiteVec.from_texts(
    texts=texts,
    embedding=embedding,
    table="state_union",
    db_file="./db/vec.db",
)

# query it
query = "What did the president say about Ketanji Brown Jackson"
data = db.as_retriever().invoke(query)
#print(data)
# from langchain_ollama.chat_models import ChatOllama
from langchain_ollama.llms import OllamaLLM
# from langchain.retrievers.multi_query import MultiQueryRetriever
# import faiss
# from langchain_community.vectorstores import FAISS 
# from langchain.storage import InMemoryByteStore

llm = OllamaLLM(base_url="http://localhost:11434", model="deepseek-llm:7b", temperature=0)
# vectordb = FAISS.from_documents(documents=docs, embedding= embedding)
# byte_store = InMemoryByteStore()

# retriever =  MultiQueryRetriever(retriever = vectordb.as_retriever(), llm = llm)
# # print results
# print(retriever.invoke(query))

from langchain_community.vectorstores import FAISS
vectordb = FAISS.from_documents(documents=docs, embedding= embedding)

from langchain_ollama import ChatOllama
from langchain.retrievers.multi_query import MultiQueryRetriever
import os
from dotenv import load_dotenv
load_dotenv()
#question = "LUA的宿主语言是什么？"
#model = ChatOllama(base_url="http://localhost:11434",model="deepseek-r1:7b",temperature=0)

# from langchain_deepseek import ChatDeepSeek
# llm = ChatDeepSeek(
#     base_url="https://api.deepseek.com",
#     model="deepseek-reasoner",
#     temperature=0,
#     max_tokens=None,
#     timeout=None,
#     max_retries=2,
#     api_key=os.getenv("DEEPSEEK_KEY"),
#     # other params...
# )

retriver_from_llm = MultiQueryRetriever.from_llm(retriever=db.as_retriever(), llm=llm)
docs = retriver_from_llm.invoke(query)

print(docs[0].page_content)