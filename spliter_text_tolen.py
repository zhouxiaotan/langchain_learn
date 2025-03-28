from langchain_text_splitters import CharacterTextSplitter
# This is a long document we can split up.
with open("C:\\WorkSpace\\LangChainLearning\\chapter2\\langchain_learn\\state_of_the_union.txt") as f:
    state_of_the_union = f.read()

# text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
#     encoding_name="cl100k_base", chunk_size=100, chunk_overlap=0
# )
from langchain_text_splitters import RecursiveCharacterTextSplitter

# text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
#     model_name="gpt-4",
#     chunk_size=100,
#     chunk_overlap=0,
# )

# texts = text_splitter.split_text(state_of_the_union)
# print(texts[0])

# from langchain_text_splitters import TokenTextSplitter

# text_splitter = TokenTextSplitter(chunk_size=10, chunk_overlap=0)

# texts = text_splitter.split_text(state_of_the_union)
# print(texts[0])

#////////////////////////////////////////////////////////////////#
# python -m spacy download en_core_web_sm
#////////////////////////////////////////////////////////////////#
# from langchain_text_splitters import SpacyTextSplitter

# text_splitter = SpacyTextSplitter(chunk_size=1000)

# texts = text_splitter.split_text(state_of_the_union)
# print(texts[0])

# pip install nltk
# This is a long document we can split up.
# import nltk
# nltk.download('punkt_tab')
# from langchain_text_splitters import NLTKTextSplitter

# text_splitter = NLTKTextSplitter(chunk_size=1000)
# texts = text_splitter.split_text(state_of_the_union)
# print(texts[0])

from transformers import GPT2TokenizerFast

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer, chunk_size=100, chunk_overlap=0
)
texts = text_splitter.split_text(state_of_the_union)
print(texts[0])