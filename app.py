import tempfile
from PIL import Image
import os


from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)

# Set the title and subtitle of the app
st.title('🦜🔗 PDF-Chat: Interact with Your PDFs in a Conversational Way')
st.subheader('Load your PDF, ask questions, and receive answers directly from the document.')

# Load the image 
image = Image.open('bot.jpeg')
st.image(image)

# Loading the Pdf file and return a temporary path for it 
st.subheader('Upload your pdf')
uploaded_file = st.file_uploader('', type=(['pdf',"tsv","csv","txt","tab","xlsx","xls"]))


if uploaded_file:
    # Read the uploaded PDF file and create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())

    # Display a text input box for the user to enter a question
    user_question = st.text_input('Ask a question about the document:', '')

    if st.button('Get Answer'):
        # Load the PDF document
        pdf_loader = PyPDFLoader(file_path=temp_file.name)
        pdf_text = pdf_loader.load()


        # Display the answer to the user
        st.write('Answer:')
        st.write("This is the answer")

        # Delete the temporary file
        os.remove(temp_file.name)
