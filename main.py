import pickle
from fastapi import FastAPI, UploadFile, File, Query
from tempfile import NamedTemporaryFile
import pdfminer
from pdfminer.high_level import extract_text
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import os
from langchain.document_loaders import UnstructuredURLLoader
from langchain.chains import RetrievalQAWithSourcesChain
from langchain import OpenAI
from langchain.callbacks import get_openai_callback
import openai
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from fastapi import FastAPI, HTTPException, Query
import certifi

import nltk

nltk.download("averaged_perceptron_tagger")

os.environ["SSL_CERT_FILE"] = certifi.where()
app = FastAPI()

def load_model_and_vector_store(url, openai_token):
    # Set the OpenAI API key
    os.environ["OPENAI_API_KEY"] = openai_token

    # Load data from the provided URL
    loaders = UnstructuredURLLoader(urls=[url])
    data = loaders.load()

    # Split the loaded text into chunks
    text_splitter = CharacterTextSplitter(separator='\n', chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(data)

    # Create OpenAI embeddings for the documents
    embeddings = OpenAIEmbeddings()

    # Build a FAISS vector store using the documents and embeddings
    vectorStore_openAI = FAISS.from_documents(docs, embeddings)

    # Save the vector store to a file
    with open("faiss_store_openai.pkl", "wb") as f:
        pickle.dump(vectorStore_openAI, f)

    # Load the vector store from the file
    with open("faiss_store_openai.pkl", "rb") as f:
        VectorStore = pickle.load(f)

    # Create a retrieval question-answering chain
    llm = OpenAI(temperature=0, model_name='text-davinci-003')
    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=VectorStore.as_retriever())

    return chain

@app.post("/qa/")
def question_answering(url: str = Query(..., alias="url"), query: str = Query(..., alias="query"), openai_token: str = Query(..., alias="openai_token")):
    # Load the model and vector store for the provided URL and OpenAI token
    qa_chain = load_model_and_vector_store(url, openai_token)

    response = qa_chain(query, return_only_outputs=True)
    return {
        "query": query,
        "answer": response["answer"],
        "source": response["sources"],
    }

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

    # Remove the temporary PDF file
    os.remove(temp_file_path)

    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8000)
