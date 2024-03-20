import io
import base64
import os
from flask import Flask, request, jsonify
from pymongo import MongoClient
import wave
import asyncio
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
from bson import json_util
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from flask_cors import CORS
from bson import ObjectId
 

 
 

os.environ.get('OPENAI_API_KEY')

app = Flask(__name__)
CORS(app)
 


# Set up MongoDB client and collection
client = MongoClient('mongodb+srv://jibran:jibranmern@clusterone.u74t8kf.mongodb.net/?retryWrites=true&w=majority')
DB_NAME = "test"
COLLECTION_NAME = "document"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_index"
MONGODB_COLLECTION = client[DB_NAME][COLLECTION_NAME]

# Initialize MongoDBAtlasVectorSearch
vector_search = MongoDBAtlasVectorSearch(
    embedding=OpenAIEmbeddings(disallowed_special=()),
    collection=MONGODB_COLLECTION,
    index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
)

# Initialize QA Retriever
qa_retriever = vector_search.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 25},
)

# Prompt Template
prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know only, don't try to make up an answer. -Importtant:Please give the response of 1 line only for all questions and response should not be greater than maximum 40 words, complete your communication withinit,. if answer is not found in context, try to get relavent answer but it should be from context, not from all over the world, you can also suggest the user that are you asking for this you are AI Banker".
 {context}
Question: "{question}"
"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Initialize QA Chain

qa = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    chain_type="stuff",
    retriever=qa_retriever,
    # return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT},
)

# List to store previous _id values
previous_ids = []

# API Endpoint for chat
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    if 'query' not in data:
        return jsonify({"error": "No 'query' provided"}), 400

    query = data['query']
    docs = qa({"query": query})

    # Convert the Document object to JSON serializable format
    return jsonify(docs["result"])

@app.route('/process_pdf', methods=['POST'])
def process_pdf():
    data = request.get_json()
    if 'pdf_base64' not in data:
        return jsonify({'error': 'No PDF data found'})

    pdf_base64 = data['pdf_base64']

    # Decode base64 to bytes
    pdf_bytes = base64.b64decode(pdf_base64)

    # Save PDF bytes to a temporary file
    temp_pdf_path = 'temp.pdf'
    with open(temp_pdf_path, 'wb') as temp_pdf:
        temp_pdf.write(pdf_bytes)

    # Load the PDF from the temporary file
    loader = PyPDFLoader(temp_pdf_path)
    pdf_data = loader.load()

    # Split the document
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(pdf_data)

    # Insert documents into MongoDB with embeddings
    try:
        vector_search_instance = MongoDBAtlasVectorSearch.from_documents(
            documents=docs,
            embedding=OpenAIEmbeddings(disallowed_special=()),
            collection=MONGODB_COLLECTION,
            index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
        )
    except Exception as e:
        # Handle specific errors
        error_message = str(e)
        return jsonify({'error': error_message}), 500

    # Retrieve the _id of the previous document
    previous_ids = []
    previous_document = MONGODB_COLLECTION.find().sort('_id', -1).limit(1)
    print(previous_document)
    
    for doc in previous_document:
      previous_id = str(doc['_id'])  # Convert ObjectId to string
      previous_ids.append(previous_id)
      print(previous_id)
  
# Remove the temporary file
    os.remove(temp_pdf_path)

# Return JSON response
    return jsonify({'message': 'PDF processed and inserted into MongoDB with embeddings!', 'previous_id': previous_ids[0]})

@app.route('/delete_all_documents', methods=['DELETE'])
def delete_all_documents():
    try:
        # Delete all documents from the 'document' collection
        result = MONGODB_COLLECTION.delete_many({})
        deleted_count = result.deleted_count
        return jsonify({'message': f'Deleted {deleted_count} documents from {COLLECTION_NAME} collection'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/display_previous_ids', methods=['GET'])
def display_previous_ids():
    # Display the list of previous _id values in the console
    print("Previous _id values:", previous_ids)
    return jsonify({'previous_ids': previous_ids})

@app.route('/', methods=['GET'])
def home():
    return "Server is running"
 



if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
