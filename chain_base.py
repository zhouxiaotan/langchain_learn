from langchain.prompts import PromptTemplate

prompt = PromptTemplate.from_template("Please write a random name related to {topic}.")


from langchain_ollama.chat_models import ChatOllama 

chat_llm = ChatOllama(base_url="http://localhost:11434",model="qwen2.5:7b",temperature=0.8)

from langchain.chains.llm import LLMChain
from langchain.chains import load_chain
#chain = LLMChain(llm= chat_llm, prompt= prompt)
chain = prompt | chat_llm

print (chain.invoke({"topic": "cats"}))
# print(output)