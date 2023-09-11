FROM python:3.10

# Add requirements file in the container
COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt

# Add source code in the container
COPY main.py ./main.py

# Add pdf in the container
COPY medical-record.pdf ./medical-record.pdf


# Set the secret environment variable
ENV OPENAI_API_KEY="YOUR_API_KEY"

# Define container entry point
ENTRYPOINT ["python", "main.py"]
