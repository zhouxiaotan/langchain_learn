from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama.chat_models import ChatOllama
from langchain_ollama.llms import OllamaLLM
from pydantic import BaseModel, Field, model_validator

model = OllamaLLM(
    base_url="http://127.0.0.1:11434", 
    #model="deepseek-r1:7b", 
    model="qwen2.5:7b",
    temperature=0.0
)

chat_model = ChatOllama(    
    base_url="http://127.0.0.1:11434", 
    #model="deepseek-r1:7b", 
    model="qwen2.5:7b",
    temperature=0.0
)

# Define your desired data structure.
class Joke(BaseModel):
    setup: str = Field(description="question to set up a joke")
    punchline: str = Field(description="answer to resolve the joke")

    # You can add custom validation logic easily with Pydantic.
    @model_validator(mode="before")
    @classmethod
    def question_ends_with_question_mark(cls, values: dict) -> dict:
        setup = values.get("setup")
        if setup and setup[-1] != "?":
            raise ValueError("Badly formed question!")
        return values


# Set up a parser + inject instructions into the prompt template.
parser = PydanticOutputParser(pydantic_object=Joke)

prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# And a query intended to prompt a language model to populate the data structure.
prompt_and_model = prompt | model
#output = prompt_and_model.invoke({"query": "Tell me a joke.response without thinking."})
#msg = parser.invoke(output)

#print(msg)

from langchain.output_parsers.json import SimpleJsonOutputParser

json_prompt = PromptTemplate.from_template(
    "Return a JSON object with an `answer` key that answers the following question: {question}"
)
json_parser = SimpleJsonOutputParser()
json_chain = json_prompt | model | json_parser

# for chunk in json_chain.stream({"question": "Who invented the microscope? response without thinking."}):
#     print(chunk, end="\n", flush=True)
#     #print("\n")

# ls = list(json_chain.stream({"question": "Who invented the microscope?"}))
# print(ls)
from typing import Iterable
from langchain_core.runnables import RunnableGenerator
from langchain_core.messages import AIMessage, AIMessageChunk
def streaming_parse(chunks: Iterable[AIMessageChunk]) -> Iterable[str]:
   
    for chunk in chunks:
        #yield chunk.content.swapcase()
        yield chunk.swapcase()
   

streaming_parse = RunnableGenerator(streaming_parse)
chain = model | streaming_parse
#print(chain.invoke("hello"))


from typing import List

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import BaseGenerationOutputParser
from langchain_core.outputs import ChatGeneration, Generation


class StrInvertCase(BaseGenerationOutputParser[str]):
    """An example parser that inverts the case of the characters in the message.

    This is an example parse shown just for demonstration purposes and to keep
    the example as simple as possible.
    """

    def parse_result(self, result: List[Generation], *, partial: bool = False) -> str:
        """Parse a list of model Generations into a specific format.

        Args:
            result: A list of Generations to be parsed. The Generations are assumed
                to be different candidate outputs for a single model input.
                Many parsers assume that only a single generation is passed it in.
                We will assert for that
            partial: Whether to allow partial results. This is used for parsers
                     that support streaming
        """
        if len(result) != 1:
            raise NotImplementedError(
                "This output parser can only be used with a single generation."
            )
        generation = result[0]
        if not isinstance(generation, ChatGeneration):
            # Say that this one only works with chat generations
            raise OutputParserException(
                "This output parser can only be used with a chat generation."
            )
        return generation.message.content.swapcase()


#chain = model | StrInvertCase() #OutputParserException
chain = chat_model | StrInvertCase()
print(chain.invoke("Tell me a short sentence about yourself"))

