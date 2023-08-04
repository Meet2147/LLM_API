# LLM_API
------------

### Introduction:
This Python application uses the LangChain library to perform question-answering on text and PDF files. The application has two main endpoints:

`/qa/`  :  This endpoint takes a URL to a text document and a query as input, and returns the answer to the query, along with the source of the answer.

`/process_pdf/ ` :  This endpoint takes a PDF file and a query as input, and returns the answer to the query, along with the source of the answer.

------------

### Requirements:
- certifi
- langchain
- nltk
- fastapi
- pdfminer.six
- pickle4
- openai
- tiktoken
- faiss-cpu
- uvicorn

Also you can find a requirements.txt file and you can install all the requirements by running the following command:

`pip install -r  requirements.txt`

------------

### Usage:
To run the application, you can use the following command:

`uvicorn main:app --port 8000`

Running the following command will start the application on your localhost on port 8000.


------------

Once the application is running, you can access the endpoints using the following URLs:
1. `/qa/`
2. `/process_pdf/`

### Contributing:

If you would like to contribute to the application, please fork the repository and submit a pull request..

------------


