import os
from os import listdir
from os.path import isfile, join
from typing import Literal, get_args
from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.schema.document import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_community.tools import WikipediaQueryRun
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Supported data sources for queries
DataSource = Literal["Wikipedia", "document"]
ALLOWED_DATA_SOURCES = get_args(DataSource)

# Load OpenAI API key from environment
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Define language model and embedding
language_model = ChatOpenAI(temperature=0, model='gpt-3.5-turbo')
embedding_function = OpenAIEmbeddings()

# Folder path for uploaded documents
document_folder = "./docs/"

def prepare_data(source: DataSource, search_query: str):
    """Prepare data for retrieval based on the specified source."""
    if source not in ALLOWED_DATA_SOURCES:
        raise ValueError(f"Source {source} is not supported in this application.")

    # Split large documents into manageable pieces
    document_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    # Handle Wikipedia data source
    if source == "Wikipedia":
        wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        data = wikipedia_tool.run(search_query)
        split_documents = [Document(page_content=sentence) for sentence in data.split('\n')]

    # Handle do data source
    else:
        file_list = [f for f in listdir(document_folder) if isfile(join(document_folder, f))]
        file_path = document_folder + file_list[0]
        print(f"Processing file: {file_path}")

        # Load and split the PDF document
        pdf_loader = PyPDFLoader(file_path)
        loaded_data = pdf_loader.load()
        split_documents = document_splitter.split_documents(loaded_data)

    # Initialize an in-memory search object for the document
    data_storage = DocArrayInMemorySearch.from_documents(documents=split_documents, embedding=embedding_function)
    return data_storage

def process_query(source: DataSource, data_storage: DocArrayInMemorySearch, search_query: str):
    """Process the query with context from data storage using language model."""
    if source not in ALLOWED_DATA_SOURCES:
        raise ValueError(f"Source {source} is not allowed.")

    # Define a custom system prompt for the query-answer chain
    context_prompt = (
        "Answer concisely using the context provided. "
        "If unsure, indicate so in your response. "
        "Max response length: three sentences. Context: {context}"
    )
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", context_prompt),
            ("human", "{input}"),
        ]
    )

    # Create the query-answer chain with retrieval and processing
    qa_chain = create_stuff_documents_chain(language_model, prompt_template)
    retrieval_chain = create_retrieval_chain(data_storage.as_retriever(), qa_chain)

    # Run the retrieval chain
    result = retrieval_chain.invoke({"input": search_query})
    return result

def get_answer(data_source: DataSource, question: str):
    """Entry point for generating answers based on the selected data source."""
    if data_source not in ALLOWED_DATA_SOURCES:
        raise ValueError(f"Data source {data_source} is not supported.")

    storage = prepare_data(data_source, question)
    answer_result = process_query(data_source, storage, question)
    
    return answer_result
