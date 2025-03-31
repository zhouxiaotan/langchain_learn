from langchain_ollama.chat_models import ChatOllama 

llm = ChatOllama(base_url="http://localhost:11434",model="qwen2.5:7b",temperature=0.8)

from langchain_core.documents import Document

documents = [
    Document(page_content="Alice has blue eyes", metadata={"title": "book_chapter_2"}),
    Document(page_content="Bob has brown eyes", metadata={"title": "book_chapter_1"}),
    Document(page_content="Charlie has green eyes", metadata={"title": "book_chapter_3"}),
]

from langchain.chains import LLMChain, MapRerankDocumentsChain
from langchain.output_parsers.regex import RegexParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI

document_variable_name = "context"
# The prompt here should take as an input variable the
# `document_variable_name`
# The actual prompt will need to be a lot more complex, this is just
# an example.
prompt_template = (
    "What color are Bob's eyes? "
    "Output both your answer and a score (1-10) of how confident "
    "you are in the format: <Answer>\nScore: <Score>.\n\n"
    "Provide no other commentary.\n\n"
    "Context: {context}"
)
output_parser = RegexParser(
    regex=r"(.*?)\nScore: (.*)",
    output_keys=["answer", "score"],
)
prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context"],
    output_parser=output_parser,
)
llm_chain = LLMChain(llm=llm, prompt=prompt)
chain = MapRerankDocumentsChain(
    llm_chain=llm_chain,
    document_variable_name=document_variable_name,
    rank_key="score",
    answer_key="answer",
)

response = chain.invoke(documents)
print(response["output_text"])