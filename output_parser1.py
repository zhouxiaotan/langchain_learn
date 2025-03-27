from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

model = ChatOllama(base_url="http://localhost:11434",model="deepseek-r1:7b",temperature=0)


# Define your desired data structure.
class Joke(BaseModel):
    setup: str = Field(description="question to set up a joke")
    punchline: str = Field(description="answer to resolve the joke")


# And a query intented to prompt a language model to populate the data structure.
joke_query = "Tell me a joke. response without thinking."

# Set up a parser + inject instructions into the prompt template.
parser = JsonOutputParser(pydantic_object=Joke)

prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = prompt | model | parser

msg = chain.invoke({"query": joke_query})
print(msg)

for s in chain.stream({"query": joke_query}):
    print(s)

parser2 = JsonOutputParser()

prompt2 = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser2.get_format_instructions()},
)

chain2 = prompt2 | model | parser2

msg = chain2.invoke({"query": joke_query})
print(msg)