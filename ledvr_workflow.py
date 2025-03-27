from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_index.llms.ollama import Ollama
import os
from dotenv import load_dotenv
load_dotenv()

# os.environ["USER_AGENT"] = "LEDVR_WORK/0.1-DEV"

loader = WebBaseLoader("https://developers.mini1.cn/wiki/luawh.html")
data = loader.load()

from langchain_ollama import ChatOllama
model = ChatOllama(base_url="http://localhost:11434",model="deepseek-r1:7b",temperature=0)


from langchain_ollama import OllamaEmbeddings

embedding = OllamaEmbeddings(
    base_url="http://localhost:11434",
    model="nomic-embed-text:latest",
    #api_key="kkkkkk",    
)

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 0)
splits = text_splitter.split_documents(data)

from langchain_community.vectorstores import FAISS
vectordb = FAISS.from_documents(documents=splits, embedding= embedding)

from langchain_ollama import ChatOllama
from langchain.retrievers.multi_query import MultiQueryRetriever

question = "LUA的宿主语言是什么？"
#model = ChatOllama(base_url="http://localhost:11434",model="deepseek-r1:7b",temperature=0)

from langchain_deepseek import ChatDeepSeek
llm = ChatDeepSeek(
    base_url="https://api.deepseek.com",
    model="deepseek-reasoner",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=os.getenv("DEEPSEEK_KEY"),
    # other params...
)

retriver_from_llm = MultiQueryRetriever.from_llm(retriever=vectordb.as_retriever(), llm=llm)
docs = retriver_from_llm.invoke(question)

print(docs[0].page_content)