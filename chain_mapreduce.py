from langchain_ollama.chat_models import ChatOllama 
from langchain_ollama.llms import OllamaLLM
from langchain_core.documents import Document
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import CharacterTextSplitter
from transformers import AutoTokenizer  # 新增导入
import os
# 配置镜像源
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"



# 初始化模型时会自动加载内置分词器
llm = OllamaLLM(model="qwen2.5:7b")

# 访问分词器（需 LangChain >=0.1.5）
tokenizer = llm._client.tokenizer  

# 使用示例
text = "Hello world"
tokens = tokenizer.encode(text)
print(tokens)  # 输出：[101, 7592, 2088, 102]


# from langchain_community.llms import Ollama
# tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")

from langchain_community.llms import Ollama

llm = Ollama(model="qwen2.5:7b")
print(llm("Hello"))  # 应返回模型响应

# 初始化分词器
# 加载兼容的分词器
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen1.5-7B",  # 或使用其他兼容模型
    trust_remote_code=True,
    timeout=300  # 5分钟超时
)

# 初始化模型时显式禁用不需要的 tokenizer
llm = ChatOllama(
    base_url="http://localhost:11434",
    model="qwen2.5:7b",
    temperature=0.8,
    tokenizer=tokenizer
)

documents = [
    Document(page_content="Apples are red", metadata={"title": "apple_book"}),
    Document(page_content="Blueberries are blue", metadata={"title": "blueberry_book"}),
    Document(page_content="Bananas are yelow", metadata={"title": "banana_book"}),
]

# # 配置文本分割器时指定 tokenizer
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  # 或使用其他合适的分词器
# text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(
#     tokenizer=tokenizer,
#     chunk_size=500,
#     chunk_overlap=50,
#     separator="\n"
# )

# Map 链保持不变
map_template = "Write a concise summary of the following: {docs}."
map_prompt = ChatPromptTemplate.from_template(map_template)
map_chain = LLMChain(llm=llm, prompt=map_prompt)

# Reduce 链保持不变
reduce_template = """
The following is a set of summaries:
{docs}
Take these and distill it into a final, consolidated summary
of the main themes.
"""
reduce_prompt = ChatPromptTemplate.from_template(reduce_template)
reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

# 关键修改：在文档链中显式传递 tokenizer
combine_documents_chain = StuffDocumentsChain(
    llm_chain=reduce_chain,
    document_variable_name="docs",
    tokenizer=tokenizer  # 显式指定
)

reduce_documents_chain = ReduceDocumentsChain(
    combine_documents_chain=combine_documents_chain,
    collapse_documents_chain=combine_documents_chain,
    token_max=1000,
    tokenizer=tokenizer  # 显式指定
)

map_reduce_chain = MapReduceDocumentsChain(
    llm_chain=map_chain,
    reduce_documents_chain=reduce_documents_chain,
    document_variable_name="docs",
    return_intermediate_steps=False,
    tokenizer=tokenizer  # 显式指定
)

# 运行链
result = map_reduce_chain.invoke(documents)
print(result["output_text"])
