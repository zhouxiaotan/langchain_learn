from langchain_community.document_loaders.csv_loader import CSVLoader

#loader = CSVLoader(file_path="./example_data/mlb_teams_2012.csv")

#data = loader.load()

#print(data)

# loader = CSVLoader(
#     file_path="./example_data/mlb_teams_2012.csv",
#     csv_args={
#         "delimiter": ",",
#         "quotechar": '"',
#         "fieldnames": ["Team", "Payroll in millions", "Wins"],
#     },
# )

# data = loader.load()

# for row in data:
#     print(row, end="\n")

# loader = CSVLoader(file_path="./example_data/mlb_teams_2012.csv", source_column="Team")

# data = loader.load()

# print(data)
from langchain_community.document_loaders.csv_loader import UnstructuredCSVLoader

loader = UnstructuredCSVLoader(
    file_path="example_data/mlb_teams_2012.csv", mode="elements"
)
docs = loader.load()

print(docs[0].metadata["text_as_html"],end="\n")