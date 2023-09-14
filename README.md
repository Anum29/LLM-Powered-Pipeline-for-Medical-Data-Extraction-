# LLM-Powered-Pipeline-for-Medical-Data-Extraction
This project is designed to extract and process information from a medical PDF document, answer specific medical-related questions based on the document content, and provide a JSON output. The project utilizes the OpenAI GPT-3.5 model, Langchain for text processing, and PyPDF2 for reading PDF files.


## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Output](#output)

## Installation

Before running the script, make sure you have the following prerequisites installed:

- Python 3.8x
- pip3


## Usage

Replace the "medical-record.pdf" with the path to the PDF file you want to process.


### Steps to run directly the python code
Within the `main.py` file, set your OpenAI API key by replacing the placeholder value OPENAI_API_KEY variable with your actual API key. Use this link to set the environment variable [here](https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety?ref=blog.streamlit.io)

- Installing dependencies:
```bash
pip3 install -r requirements.txt
```
- Running the python code:
```bash
python3 main.py
```


- Running the streamit app code:
```bash
streamlit run app.py
```

### Steps to run the python code withing a Docker container

- Build the image:
```bash
docker build -t docker-llm-data-pipeline:latest .
```

- Run the Docker container with the secret environment variable:
```bash
docker run -e OPENAI_API_KEY="YOUR_API_KEY" docker-llm-data-pipeline:latest
```

The script will perform the following tasks:

1. Extract text from the PDF document(s).
2. Split the text into manageable chunks and calculate embeddings.
3. Create a document search instance using FAISS.
4. Process a list of predefined queries to extract specific information.
5. Save the results in a JSON file named patient_info.json.
Please note that the script introduces a sleep of 5 seconds between queries to avoid making requests too quickly. You can adjust this sleep time as needed.


## Output
The final output of the script will be a JSON object with appropriate structure to represent the extracted information and answers to the medical questions.
