# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

ADD pdf_query_langchain.py 

ENTRYPOINT [ "python", "./pdf_query_langchain.py"]
