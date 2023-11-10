# Import os to set API key
import os
import tempfile
from PIL import Image, ImageOps
import json
import time
import string
from PyPDF2 import PdfReader
from main import read_pdf
from main import text_embedding
from main import search_documents
from main import query_result
from main import process_queries
from main import justification_and_confidence_score


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
