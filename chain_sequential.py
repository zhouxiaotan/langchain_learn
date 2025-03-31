from langchain_ollama.chat_models import ChatOllama 

llm = ChatOllama(base_url="http://localhost:11434",model="qwen2.5:7b",temperature=0.8)

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
# This is an LLMChain to write a synopsis given a title of a play.
template = """You are a playwright. Given the title of play, it is your job to write a synopsis for that title.

Title: {title}
Playwright: This is a synopsis for the above play:"""
prompt_template = PromptTemplate(input_variables=["title"], template=template)
synopsis_chain = prompt_template | llm

# This is an LLMChain to write a review of a play given a synopsis.
template = """You are a play critic from the New York Times. Given the synopsis of play, it is your job to write a review for that play.

Play Synopsis:
{synopsis}
Review from a New York Times play critic of the above play:"""
prompt_template = PromptTemplate(input_variables=["synopsis"], template=template)
review_chain = prompt_template | llm

from langchain_core.runnables import RunnablePassthrough

overall_chain = (
    RunnablePassthrough()  # Passes input through
    | synopsis_chain
    | review_chain
)

review = overall_chain.invoke("Tragedy at sunset on the beach")

print(review)