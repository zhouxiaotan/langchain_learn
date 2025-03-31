from langchain_ollama.chat_models import ChatOllama 

llm = ChatOllama(base_url="http://localhost:11434",model="qwen2.5:7b",temperature=0.8)

from langchain_core.documents import Document

documents = [
    Document(page_content="Apples are red", metadata={"title": "apple_book"}),
    Document(page_content="Blueberries are blue", metadata={"title": "blueberry_book"}),
    Document(page_content="Bananas are yelow", metadata={"title": "banana_book"}),
]
from langchain.chains import LLMChain, RefineDocumentsChain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI

# This controls how each document will be formatted. Specifically,
# it will be passed to `format_document` - see that function for more
# details.
document_prompt = PromptTemplate(
    input_variables=["page_content"], template="{page_content}"
)
document_variable_name = "context"
# The prompt here should take as an input variable the
# `document_variable_name`
summarize_prompt = ChatPromptTemplate(
    [
        ("human", "Write a concise summary of the following: {context}"),
    ]
)
initial_llm_chain = LLMChain(llm=llm, prompt=summarize_prompt)
initial_response_name = "existing_answer"
# The prompt here should take as an input variable the
# `document_variable_name` as well as `initial_response_name`
refine_template = """
Produce a final summary.

Existing summary up to this point:
{existing_answer}

New context:
------------
{context}
------------

Given the new context, refine the original summary.
"""
refine_prompt = ChatPromptTemplate([("human", refine_template)])
refine_llm_chain = LLMChain(llm=llm, prompt=refine_prompt)
chain = RefineDocumentsChain(
    initial_llm_chain=initial_llm_chain,
    refine_llm_chain=refine_llm_chain,
    document_prompt=document_prompt,
    document_variable_name=document_variable_name,
    initial_response_name=initial_response_name,
)

# result = chain.invoke(documents)
# print(result["output_text"])


import operator
from typing import List, Literal, TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph

#llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Initial summary
summarize_prompt = ChatPromptTemplate(
    [
        ("human", "Write a concise summary of the following: {context}"),
    ]
)
initial_summary_chain = summarize_prompt | llm | StrOutputParser()

# Refining the summary with new docs
refine_template = """
Produce a final summary.

Existing summary up to this point:
{existing_answer}

New context:
------------
{context}
------------

Given the new context, refine the original summary.
"""
refine_prompt = ChatPromptTemplate([("human", refine_template)])

refine_summary_chain = refine_prompt | llm | StrOutputParser()


# For LangGraph, we will define the state of the graph to hold the query,
# destination, and final answer.
class State(TypedDict):
    contents: List[str]
    index: int
    summary: str


# We define functions for each node, including a node that generates
# the initial summary:
async def generate_initial_summary(state: State, config: RunnableConfig):
    summary = await initial_summary_chain.ainvoke(
        state["contents"][0],
        config,
    )
    return {"summary": summary, "index": 1}


# And a node that refines the summary based on the next document
async def refine_summary(state: State, config: RunnableConfig):
    content = state["contents"][state["index"]]
    summary = await refine_summary_chain.ainvoke(
        {"existing_answer": state["summary"], "context": content},
        config,
    )

    return {"summary": summary, "index": state["index"] + 1}


# Here we implement logic to either exit the application or refine
# the summary.
def should_refine(state: State) -> Literal["refine_summary", END]:
    if state["index"] >= len(state["contents"]):
        return END
    else:
        return "refine_summary"


graph = StateGraph(State)
graph.add_node("generate_initial_summary", generate_initial_summary)
graph.add_node("refine_summary", refine_summary)

graph.add_edge(START, "generate_initial_summary")
graph.add_conditional_edges("generate_initial_summary", should_refine)
graph.add_conditional_edges("refine_summary", should_refine)
app = graph.compile()

from IPython.display import Image

image = Image(app.get_graph().draw_mermaid_png())
save_path = "refine_graph.png"
# 创建文件对象并写入数据
with open(save_path, "wb") as file:
    file.write(image.data)

import asyncio

async def process_documents():
    async for step in app.astream(
        {"contents": [doc.page_content for doc in documents]},
        stream_mode="values",
    ):
        if summary := step.get("summary"):
            print(summary)

# Run the async function
asyncio.run(process_documents())
