from langchain_ollama.chat_models import ChatOllama 

llm = ChatOllama(base_url="http://localhost:11434",model="qwen2.5:7b",temperature=0.8)

from operator import itemgetter
from typing import Literal
from typing_extensions import TypedDict
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

prompt_1 = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert on animals."),
        ("human", "{query}"),
    ]
)
prompt_2 = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert on vegetables."),
        ("human", "{query}"),
    ]
)
chain_1 = prompt_1 | llm | StrOutputParser()
chain_2 = prompt_2 | llm | StrOutputParser()
route_system = "Route the user's query to either the animal or vegetable expert."
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", route_system),
        ("human", "{query}"),
    ]
)
class RouteQuery(TypedDict):
    """Route query to destination."""
    destination: Literal["animal", "vegetable"]
route_chain = (
    route_prompt
    | llm.with_structured_output(RouteQuery)
    | itemgetter("destination")
)
chain = {
    "destination": route_chain,  # "animal" or "vegetable"
    "query": lambda x: x["query"],  # pass through input query
} | RunnableLambda(
    # if animal, chain_1. otherwise, chain_2.
    lambda x: chain_1 if x["destination"] == "animal" else chain_2,
)
msg = chain.invoke({"query": "what color are carrots"}, debug=True)
print(msg)