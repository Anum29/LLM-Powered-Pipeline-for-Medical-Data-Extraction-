# -*- coding: utf-8 -*-
"""LLM Powered Pipeline for Medical Data Extraction
"""


import os
import json
import time
import string

from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

from typing_extensions import Concatenate

OPENAI_MODEL = "gpt-3.5-turbo"
#OPENAI_API_KEY =  'sk-XXXX' Add your API key here if running script using python

#os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')


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
    return results

def justification_and_confidence_score(data_dict, document_text):
    """
    Generate justifications and confidence scores for each answer based on the document text.

    Args:
        data_dict (dict): The dictionary containing answers and their values.
        document_text (str): The text content of the document.

    Returns:
        dict: A new dictionary with justifications and confidence scores.
    """
    result_dict = {}  # Initialize a dictionary to store results

    for key, value in data_dict.items():
        # Initialize justification and confidence variables
        justification = None
        confidence = None

        # Convert the value to a string
        value_str = str(value) + "-" + str(key)

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

        # Add the sub-dictionary to the result dictionary
        result_dict[key] = sub_dict

    return result_dict

def query_rewriting(queries):
    """
    Rewrite the queries according to specified rules and generate a new dictionary.

    Args:
        queries (dict): The original dictionary of queries.

    Returns:
        dict: A new dictionary of rewritten queries.
    """
    rewritten_queries = {}  # Initialize a dictionary to store rewritten queries

    for key, query in queries.items():
        # Initialize a list to store modifications to the query
        modifications = []

        # Check for specific keywords in the query and apply modifications
        if "Comment" in query:
            query = query + ". Elaborate."
        elif "chief" in query:
            query = query + " in one word"
        elif "plan" in query:
            query = query + ". Answer in bullets"
        elif "list" in query:
            query = query + ". Answer in bullets"
        elif "?" in query:
            query = query + " Answer yes or no"


        # Store the modified query in the new dictionary
        rewritten_queries[key] = query

    return rewritten_queries

if __name__ == '__main__':

    # provide the path of  pdf file/files.
    pdfreader = PdfReader('medical-record.pdf')
    raw_text = read_pdf(pdfreader)
    texts, embeddings = text_embedding(raw_text)

    document_search = search_documents(texts, embeddings)


    # Original queries
    queries = {
        "Chief Complaint": "Patient’s chief complaint",
        "Treatment Plan": "Suggested treatment plan",
        "Allergies": "A list of allergies the patient has",
        "Medications": "A list of medications the patient is taking, with any known side-effects",
        "Family history of hypertension": "Does the patient have a family history of hypertension?",
        "Family history of colon cancer": "Does the patient have a family history of colon cancer?",
        "Family history of colon cancer": "Does the patient have a family history of colon cancer?",
        "Red blood per rectum" : "Has the patient experienced minimal bright red blood per rectum?",
        }

    # Rewrite the queries
    rewritten_queries = query_rewriting(queries)

    # Display the rewritten queries
    for key, query in rewritten_queries.items():
        print(f"{key}: {query}")

    result = process_queries(document_search, rewritten_queries)

    # Generate justifications and confidence scores
    output = justification_and_confidence_score(result, raw_text)

    print(output)

    # Display the results
    for key, value in output.items():
        print(f"{key}:")
        print(f"  Answer: {value['Answer']}")
        print(f"  Justification: {value['Justification']}")
        print(f"  Confidence: {value['Confidence']}")
        print()


    query = "Comment on treatment plan. Is the treatment plan correct according to clinical accuracy? Consider family history of hypertension, colon cancer, red blood cell per rectum"
    query_result(document_search, query)

    result["Comment on treatment plan"] = query_result(document_search, query)

    print(result)

    # Save the results in a JSON file
    with open("patient_info.json", "w") as json_file:
      json.dump(result, json_file, indent=4)
      print("Patient information has been retrieved and saved in 'patient_info.json'.")

