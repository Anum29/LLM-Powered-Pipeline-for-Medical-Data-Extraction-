# -*- coding: utf-8 -*-
"""LLM Powered Pipeline for Medical Data Extraction"
"""

import os
import json
import time

from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

from typing_extensions import Concatenate

OPENAI_MODEL = "gpt-3.5-turbo"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

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

        # Introduce a sleep to avoid making requests too quickly
        time.sleep(5)  # Sleep for 5 seconds (adjust as needed)

    # Save the results in a JSON file
    with open("patient_info.json", "w") as json_file:
        json.dump(results, json_file, indent=4)

    print("Patient information has been retrieved and saved in 'patient_info.json'.")
    return results

def generate_justifications_and_confidence(document_text, extracted_data):
    """
    Generate justifications and confidence scores for each answer based on the extracted data.

    Args:
        document_text (str): The text content of the PDF document.
        extracted_data (dict): The extracted data in JSON format.

    Returns:
        dict: A dictionary containing justifications and confidence scores for each answer.
    """
    justifications_and_confidence = {}  # Initialize a dictionary to store results

    for key, answer in extracted_data.items():
        justification = None
        confidence = None

        # Check if the answer contains "Yes" or "No" for boolean questions
        if answer.strip().lower() == "yes" or answer.strip().lower() == "no":
            # If the answer is "Yes" or "No," set high confidence
            confidence = "10/10"  # High confidence

            # Generate justifications for "Yes" or "No" answers based on document content
            if key == "Family history of hypertension":
                if "Hypertensive disorder" in document_text:
                    justification = "Hypertensive disorder is mentioned in the document."
                else:
                    justification = "Hypertensive disorder is not explicitly mentioned in the document."
            elif key == "Family history of colon cancer":
                if "Colon" in document_text:
                    justification = "Colon cancer is mentioned in the document."
                else:
                    justification = "Colon cancer is not mentioned in the document."
            elif key == "Red blood per rectum":
                if "minimal bright red blood per rectum" in document_text:
                    justification = "There is evidence of minimal bright red blood per rectum in the document."
                else:
                    justification = "There is no evidence of minimal bright red blood per rectum in the document."

        # Store justification and confidence in the results dictionary
        if justification:
            justifications_and_confidence[f"{key} Justification"] = justification
        if confidence:
            justifications_and_confidence[f"{key} Confidence"] = confidence

    return justifications_and_confidence

# provide the path of  pdf file/files.
pdfreader = PdfReader('medical-record.pdf')
raw_text = read_pdf(pdfreader)
texts, embeddings = text_embedding(raw_text)

document_search = search_documents(texts, embeddings)

query = "Patient’s chief complaint in one word"
query_result(document_search, query)

# Create query prompts for each piece of information
queries = {
    "Chief Complaint": "Patient’s chief complaint in one word",
    "Treatment Plan": "Suggested treatment plan. Answer in bullets",
    "Allergies": "A list of allergies",
    "Medications": "A list of medications the patient is taking, with any known side-effects. Answer in bullets",
    "Family history of hypertension": "Does the patient have a family history of hypertension?Answer yes or no",
    "Family history of colon cancer": "Does the patient have a family history of colon cancer? Answer yes or no",
    "Family history of colon cancer": "Does the patient have a family history of colon cancer? Answer yes or no",
    "Red blood per rectum" : "Has the patient experienced minimal bright red blood per rectum? Answer yes or no",
    "Comment on treatment plan":"Is the treatment plan correct according to clinical accuracy? Consider family history of hypertension, colon cancer, red blood cell per rectum"

}

result = process_queries(document_search, queries)

print(result)

# Example usage:
document_text = raw_text
extracted_data = {
    "Family history of hypertension": result["Family history of hypertension"],
    "Family history of colon cancer": result["Family history of colon cancer"],
    "Red blood per rectum": result["Red blood per rectum"],
}

justifications_and_confidence = generate_justifications_and_confidence(document_text, extracted_data)
print(justifications_and_confidence)
