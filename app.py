# Import os to set API key
import os
import tempfile
from PIL import Image, ImageOps
import json
import time
import string
from PyPDF2 import PdfReader

# Import OpenAI as main LLM service
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
# Bring in streamlit for UI/app interface
import streamlit as st
# Import PDF document loaders...there's other ones as well!
from langchain.document_loaders import PyPDFLoader
# Import chroma as the vector store 
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from typing_extensions import Concatenate

# Import vector store stuff
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)


def read_pdf(pdfreader):
    """
    Read text from a PDF file and return the concatenated text content.

    Args:
        pdfreader (PdfReader): An instance of PyPDF2's PdfReader class.

    Returns:
        str: The extracted text content from the PDF.
    """
    raw_text = ''
    for i, page in enumerate(pdfreader.pages):
        content = page.extract_text()
        if content:
            raw_text += content
    return raw_text

def text_embedding(raw_text):
    """
    Split the input raw text and calculate embeddings.

    Args:
        raw_text (str): The raw text to be processed.

    Returns:
        list: List of split texts.
        OpenAIEmbeddings: Embeddings object.
    """
    text_splitter = CharacterTextSplitter(
        separator=",",
        chunk_size=800,
        chunk_overlap=200,
        length_function=len,
    )

    # Split the text
    texts = text_splitter.split_text(raw_text)

    # Create an OpenAIEmbeddings object
    embeddings = OpenAIEmbeddings()

    # Return the results
    return texts, embeddings

# Function to create a document search instance
def search_documents(texts, embeddings):
    """
    Create a document search instance using FAISS.

    Args:
        texts (list): List of texts/documents to be indexed.
        embeddings (OpenAIEmbeddings): Embeddings object.

    Returns:
        FAISS: A document search instance.
    """
    # Create a document search instance using FAISS and the provided texts and embeddings
    document_search = FAISS.from_texts(texts, embeddings)
    return document_search

# Define a function to search for information in a document
def query_result(document_search, query):
    """
    Perform a query on a document and return the result using an OpenAI QA chain.

    Args:
        document_search: An instance of the document search module.
        query (str): The query to search for in the document.

    Returns:
        dict: A dictionary containing the result from the OpenAI QA chain.
    """
    # Load the OpenAI QA chain
    chain = load_qa_chain(OpenAI(), chain_type="stuff")

    # Use the document search module to find relevant documents
    docs = document_search.similarity_search(query)

    # Run the OpenAI QA chain using the retrieved documents and query
    result = chain.run(input_documents=docs, question=query)

    return result

def process_queries(document_search, queries):
    """
    Process a list of queries and store the results in a JSON file.

    Args:
        document_search (function): A function that searches for information.
        queries (dict): A dictionary containing queries.

    Returns:
        dict: A dictionary containing the results.
    """
    results = {}  # Initialize a dictionary to store the results
    index = 0

    # Retrieve and store information for each query
    for key, query in queries.items():
        output = query_result(document_search, query)

        # Check if the output contains bullet points
        if "\n" in output:
            # Split the output by bullet points
            bullet_points = output.split('\n')
            bullet_points = [x for x in bullet_points if len(x) > 1]
            sub_dict = {}  # Initialize a sub-dictionary for this query

            # Populate the sub-dictionary with bullet points as values indexed by bullet count
            for i, bullet_point in enumerate(bullet_points, start=1):
                sub_dict[f"{i}"] = bullet_point.strip()

            # Store the sub-dictionary in the results dictionary
            results[key] = sub_dict
        else:
            # If there are no bullet points, store the output directly
            results[key] = output

    return results


def justification_and_confidence_score(query, document_text):
    """
    Generate justifications and confidence scores for each answer based on the document text.

    Args:
        query: The query
        document_text (str): The text content of the document.

    Returns:
        dict: A new dictionary with justifications and confidence scores.
    """
    result_dict = {}  # Initialize a dictionary to store results

    # Initialize justification and confidence variables
    justification = None
    confidence = None

    # Convert the value to a string
    value_str = str(query)

    # Convert both the document text and answer to lowercase for case-insensitive comparison
    document_text_lower = document_text.lower()
    value_lower = value_str.lower()

    # Remove punctuation from document text and value
    translator = str.maketrans('', '', string.punctuation)
    document_text_cleaned = document_text_lower.translate(translator)
    value_cleaned = value_lower.translate(translator)

    # Split the cleaned text into words
    document_words = set(document_text_cleaned.split())
    value_words = set(value_cleaned.split())

    # Calculate the intersection of words
    common_words = document_words.intersection(value_words)

    # Determine confidence score based on the number of common words
    if len(common_words) >= 1:
        confidence = "10/10"
    else:
        confidence = "0"

    # Generate justification by finding matching text in document_text
    if len(common_words) > 0:
        # Find the first matching word
        matching_word = list(common_words)[0]

        # Find the sentence containing the matching word
        sentences = document_text.split('.')
        for sentence in sentences:
            if matching_word in sentence.lower():
                justification = f"The answer '{value_str}' is justified by the following text: '{sentence.strip()}'."
                break

    # Create a sub-dictionary for this answer
    sub_dict = {
        "Answer": value_str,
        "Justification": justification,
        "Confidence": confidence
    }

    return sub_dict


# Set the title and subtitle of the app
st.title('🧑🏻‍⚕️. 👩🏻‍⚕️.🔗 Medical-Record-PDF-Chat: Interact with Your PDFs in a Conversational Way')
st.subheader('Load your PDF, ask questions, and receive answers directly from the document.')

# Open the image
image = Image.open('pic.jpg')

st.image(image)

# Loading the Pdf file and return a temporary path for it 
st.subheader('Upload your pdf')
uploaded_file = st.file_uploader('', type=(['pdf',"tsv","csv","txt","tab","xlsx","xls"]))

temp_file_path = os.getcwd()        
while uploaded_file is None:
    x = 1
        
if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    temp_dir = tempfile.TemporaryDirectory()
    temp_file_path = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(uploaded_file.read())

    st.write("Full path of the uploaded file:", temp_file_path)

# Set APIkey for OpenAI Service
# Can sub this out for other LLM providers
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

# Create instance of OpenAI LLM
llm = OpenAI(temperature=0.1, verbose=True)
embeddings = OpenAIEmbeddings()

pdfreader = PdfReader(uploaded_file)
raw_text = read_pdf(pdfreader)

# Create and load PDF Loader
loader = PyPDFLoader(temp_file_path)
# Split pages from pdf 
pages = loader.load_and_split()

# Load documents into vector database aka ChromaDB
store = Chroma.from_documents(pages, embeddings, collection_name='Pdf')

# Create vectorstore info object
vectorstore_info = VectorStoreInfo(
    name="Pdf",
    description=" A pdf file to answer your questions",
    vectorstore=store
)
# Convert the document store into a langchain toolkit
toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)

# Add the toolkit to an end-to-end LC
agent_executor = create_vectorstore_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True
)

# Initialize a dictionary to store query-response pairs
query_responses = {}

# Create a text input box for the user
prompt = st.text_input('Input your prompt here')

# If the user hits enter
if prompt:

    texts, embeddings = text_embedding(raw_text)
    document_search = search_documents(texts, embeddings)
    query_result(document_search, prompt)

    # Then pass the prompt to the LLM
    response = query_result(document_search, prompt)
    
    # Store the query and its corresponding response in the dictionary
    query_responses[prompt] = response

    # ...and write it out to the screen
    st.write(response)

    # With a streamlit expander  
    with st.expander('Document Similarity Search'):
        # Find the relevant pages
        search = store.similarity_search_with_score(response) 
        # Write out the first 
        st.write(search[0][0].page_content)

         # Generate justifications and confidence scores
        output = justification_and_confidence_score(response, raw_text)

        st.write(output)

# Display the stored query-response pairs
st.subheader('Query-Response Pairs:')
st.write(query_responses)
