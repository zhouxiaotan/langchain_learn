from langchain_ollama.chat_models import ChatOllama 

llm = ChatOllama(base_url="http://localhost:11434",model="qwen2.5:7b",temperature=0.8)

from langchain_core.documents import Document

documents = [
    Document(page_content="Apples are red", metadata={"title": "apple_book"}),
    Document(page_content="Blueberries are blue", metadata={"title": "blueberry_book"}),
    Document(page_content="Bananas are yelow", metadata={"title": "banana_book"}),
]

from langchain.chains import LLMChain, StuffDocumentsChain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

# This controls how each document will be formatted. Specifically,
# it will be passed to `format_document` - see that function for more
# details.
document_prompt = PromptTemplate(
    input_variables=["page_content"], template="{page_content}"
)
document_variable_name = "context"
# The prompt here should take as an input variable the
# `document_variable_name`
prompt = ChatPromptTemplate.from_template("Summarize this content: {context}")

llm_chain = LLMChain(prompt = prompt, llm = llm)
chain = StuffDocumentsChain(
    llm_chain=llm_chain,
    document_prompt=document_prompt,
    document_variable_name=document_variable_name,
)

result = chain.invoke(documents)
print(result["output_text"])

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("Summarize this content: {context}")
chain = create_stuff_documents_chain(llm, prompt)

result = chain.invoke({"context": documents})
print(result)