# LLM-Powered-Pipeline-for-Medical-Data-Extraction-
This project is designed to extract and process information from a medical PDF document, answer specific medical-related questions based on the document content, and provide a JSON output. The project utilizes the OpenAI GPT-3.5 model, Langchain for text processing, and PyPDF2 for reading PDF files.


## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)

## Introduction

Provide a brief introduction to your project. Explain what it does and why it's useful or interesting.

## Installation

Before running the script, make sure you have the following prerequisites installed:

- Python 3.x
- pip3
You also need to install the required Python packages by running:

```shell
pip3 install -r requirements.txt
```

## Usage
1. Replace the "medical-record.pdf" with the path to the PDF file you want to process.

2. Set your OpenAI API key by replacing the placeholder value 'sk-XXX in the OPENAI_API_KEY variable with your actual API key.

3. Run the script using Python:

The script will perform the following tasks:
```bash
python3 PdfQueryLangchain.ipynb
```

1. Read the text content from the provided PDF file.
2. Split the text into smaller chunks and calculate embeddings.
3. Create a document search instance using FAISS.
4. Process a list of queries based on the document content and store the results in a JSON file named "patient_info.json".
5. The JSON output will contain information related to the patient's chief complaint, treatment plan, allergies, medications, and answers to specific medical questions.

Please note that the script introduces a sleep of 5 seconds between queries to avoid making requests too quickly. You can adjust this sleep time as needed.


Output
The final output of the script will be a JSON object with appropriate structure to represent the extracted information and answers to the medical questions.
