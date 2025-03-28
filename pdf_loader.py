from langchain_community.document_loaders import UnstructuredPDFLoader
# import matplotlib as mpl
# print(mpl.get_cachedir())  # Show cache location

file_path = "./example_data/layout-parser-paper.pdf"
#loader = UnstructuredPDFLoader(file_path)

# loader = UnstructuredPDFLoader(file_path, mode="elements")

# docs = loader.load()
# print(docs[0])


# from langchain_community.document_loaders import OnlinePDFLoader

# loader = OnlinePDFLoader("https://arxiv.org/pdf/2302.03803.pdf")
# data = loader.load()
# print(data[0])

from langchain_community.document_loaders import PDFMinerLoader

# file_path = "./example_data/layout-parser-paper.pdf"
# loader = PDFMinerLoader(file_path)

# docs = loader.load()
# import pprint

# pprint.pp(docs[0].page_content)


from langchain_community.document_loaders.parsers import RapidOCRBlobParser

loader = PDFMinerLoader(
    "./example_data/layout-parser-paper.pdf",
    mode="page",
    images_inner_format="markdown-img",
    images_parser=RapidOCRBlobParser(),
)
docs = loader.load()

print(docs[5].page_content)