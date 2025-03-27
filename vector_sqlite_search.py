from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import SQLiteVec
from langchain_text_splitters import CharacterTextSplitter

# load the document and split it into chunks
# loader = TextLoader("C:\\WorkSpace\\LangChainLearning\\chapter2\\langchain_learn\\state_of_the_union.txt")
# documents = loader.load()

# # split it into chunks
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# docs = text_splitter.split_documents(documents)
# texts = [doc.page_content for doc in docs]


# create the open-source embedding function
from langchain_ollama.embeddings import OllamaEmbeddings

embedding = OllamaEmbeddings(base_url="http://localhost:11434",model="nomic-embed-text:latest")
connection = SQLiteVec.create_connection(db_file="C:\\WorkSpace\\LangChainLearning\\chapter2\\langchain_learn\\vec.db")

db1 = SQLiteVec(
    table="state_union", embedding=embedding, connection=connection
)

#db1.add_texts(["Ketanji Brown Jackson is awesome"])
# query it again
query = "What did the president say about Ketanji Brown Jackson"
data = db1.similarity_search(query)

# print results
print(data[0].page_content)