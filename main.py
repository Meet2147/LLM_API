import pickle
from fastapi import FastAPI, UploadFile, File, Query
from tempfile import NamedTemporaryFile
import pdfminer
from pdfminer.high_level import extract_text
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader
from langchain import OpenAI
from langchain.callbacks import get_openai_callback
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from fastapi import FastAPI, HTTPException, Query
from pathlib import Path
import certifi
import os
import nltk
import pandas as pd


nltk.download("averaged_perceptron_tagger")

os.environ["SSL_CERT_FILE"] = certifi.where()
app = FastAPI()


TEMP_FOLDER = Path("./temp")
TEMP_FOLDER.mkdir(exist_ok=True)  # Ensure the temp folder exists

# Define file handling and processing functions
def convert_excel_to_csv(excel_file):
    df = pd.read_excel(excel_file.file, engine='openpyxl')
    temp_csv_path = TEMP_FOLDER / f"{excel_file.filename}.csv"
    df.to_csv(temp_csv_path, index=False, encoding='utf-8')
    return temp_csv_path

def save_uploaded_file(uploaded_file, suffix):
    temp_file_path = TEMP_FOLDER / f"{uploaded_file.filename}{suffix}"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.file.read())
    return temp_file_path

def process_csv(file_path):
    loader = CSVLoader(str(file_path))  # Convert path to string
    data = loader.load()
    return data

def process_powerpoint(file_path):
    loader = UnstructuredPowerPointLoader(str(file_path))  # Convert path to string
    data = loader.load()
    return data


def process_file(uploaded_file: UploadFile = File(...), openai_api_key: str = File(...)):
    # Determine the file type and process accordingly
    if uploaded_file.filename.endswith('.xlsx'):
        tmp_file_path = convert_excel_to_csv(uploaded_file)
        data = process_csv(tmp_file_path)
    elif uploaded_file.filename.endswith('.csv'):
        tmp_file_path = save_uploaded_file(uploaded_file, ".csv")
        data = process_csv(tmp_file_path)
    elif uploaded_file.filename.endswith('.pptx'):
        tmp_file_path = save_uploaded_file(uploaded_file, ".pptx")
        data = process_powerpoint(tmp_file_path)
    
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    # Process the data to generate embeddings and a retrieval chain
    
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    vectors = FAISS.from_documents(data, embeddings)
    chain = RetrievalQAWithSourcesChain.from_llm(
        llm=ChatOpenAI(temperature=0.0, model_name='gpt-3.5-turbo', openai_api_key=openai_api_key),
        retriever=vectors.as_retriever()
    )
    return chain 

@app.post("/process_pdf")
async def process_pdf_endpoint(
    file: UploadFile = File(...),
    query: str = Query(..., alias="query"),
    openai_token: str = Query(..., description="OpenAI API key"),
):
    # Set the OpenAI API key
    os.environ["OPENAI_API_KEY"] = openai_token

    # Save the uploaded PDF file temporarily
    with NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(await file.read())
        temp_file_path = temp_file.name

    # Process the PDF and extract raw text
    raw_text = extract_text(temp_file_path)

    # Split the text into smaller chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_text(raw_text)

    # Download embeddings from OpenAI
    embeddings = OpenAIEmbeddings()

    # Create the document search index
    docsearch = FAISS.from_texts(texts, embeddings)

    # Load the question-answering chain
    chain = load_qa_chain(OpenAI(), chain_type="stuff")
    
    docs = docsearch.similarity_search(query)
    response = chain.run(input_documents=docs, question=query)
    os.remove(temp_file_path)
    return{
            "query": query,
            "answer": response,
        }
    

@app.post("/process_excel/")
async def process_pdf_endpoint(
    file: UploadFile = File(...),
    query: str = Query(..., alias="query"),
    openai_token: str = Query(..., description="OpenAI API key"),
):
    # Load the model and vector store for the provided URL and OpenAI token
    qa_chain = process_file(file, openai_token)

    response = qa_chain(query, return_only_outputs=True)
    print(response)
    return{
            "query": query,
            "answer": response['answer'],
        }

@app.post("/process_csv/")
async def process_pdf_endpoint(
    file: UploadFile = File(...),
    query: str = Query(..., alias="query"),
    openai_token: str = Query(..., description="OpenAI API key"),
):
    # Load the model and vector store for the provided URL and OpenAI token
    qa_chain = process_file(file, openai_token)

    response = qa_chain(query, return_only_outputs=True)
    print(response)
    return{
            "query": query,
            "answer": response['answer'],
        }

@app.post("/process_ppt/")
async def process_pdf_endpoint(
    file: UploadFile = File(...),
    query: str = Query(..., alias="query"),
    openai_token: str = Query(..., description="OpenAI API key"),
):
    # Load the model and vector store for the provided URL and OpenAI token
    qa_chain = process_file(file, openai_token)

    response = qa_chain(query, return_only_outputs=True)
    print(response)
    return{
            "query": query,
            "answer": response['answer'],
        }
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8000)
