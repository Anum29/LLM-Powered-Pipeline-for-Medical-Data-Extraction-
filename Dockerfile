# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables for OpenAI API key
ENV OPENAI_API_KEY=YOUR_OPENAI_API_KEY

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the app runs on (optional)
# EXPOSE 80

# Define the command to run your script
CMD ["python", "pdf_query_langchain.py"]
