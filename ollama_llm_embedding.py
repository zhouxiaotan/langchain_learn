from llama_index.llms.openai import OpenAI

from llama_index.llms.ollama import Ollama

# llm = Ollama(
#     base_url="http://localhost:11434",model="deepseek-r1:7b",request_timeout=300
# )

# from llama_index.core.prompts import PromptTemplate

# prompt = PromptTemplate("Please write a random name related to {topic}.")
# output = llm.predict(prompt, topic="cats")
# print(output)

from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction

embed_model= OllamaEmbeddingFunction(url="http://localhost:11434/api/embeddings", model_name="nomic-embed-text:latest")

print(embed_model("this is a test"))


