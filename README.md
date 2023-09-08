# LLM-Powered-Pipeline-for-Medical-Data-Extraction
This project is designed to extract and process information from a medical PDF document, answer specific medical-related questions based on the document content, and provide a JSON output. The project utilizes the OpenAI GPT-3.5 model, Langchain for text processing, and PyPDF2 for reading PDF files.


## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Output](#output)

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
### Running through python script
1. Replace the "medical-record.pdf" with the path to the PDF file you want to process.

2. Set your OpenAI API key by replacing the placeholder value 'sk-XXX in the OPENAI_API_KEY variable with your actual API key.

3. Run the script using Python:

The script will perform the following tasks:

```bash

python3 pdf_query_langchain.py

```

The script will perform the following tasks:

1. Extract text from the PDF document(s).
2. Split the text into manageable chunks and calculate embeddings.
3. Create a document search instance using FAISS.
4. Process a list of predefined queries to extract specific information.
5. Save the results in a JSON file named patient_info.json.
Please note that the script introduces a sleep of 5 seconds between queries to avoid making requests too quickly. You can adjust this sleep time as needed.

#### Running through docker
To run using the docker:
```bash
docker build -t pdf-query-langchain
```

```bash
docker run -e OPENAI_API_KEY=YOUR_API_KEY pdf-query-langchain
```


## Output
The final output of the script will be a JSON object with appropriate structure to represent the extracted information and answers to the medical questions.
