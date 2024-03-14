import streamlit as st
import pdfplumber

from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv
from langchain import hub
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

def generate_response(file_name, openai_api_key, query_text):
    # Load document if file is uploaded
    # if file_name is not None:
    # Path to the local document file
    
    current_directory = os.getcwd() 

    # Path to the local document file in the same folder
    # local_document_path = os.path.join(current_directory , 'docs/' + file_name)

    # Read the contents of the local document
    text = ''
    for file in file_name:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text += page.extract_text()

    # Assign the local document content to the 'documents' list
    documents = [text]

    # Split documents into chunks
    # Initialize the CharacterTextSplitter with best practices
    # Optimal chunk size and overlap for context preservation
    chunk_size = 1000  # Adjust based on your document and model's token limit #TODO: change this at the end
    chunk_overlap = 50  # Adjust based on average sentence length in your text
    separator = "."  # Adjust based on your document's structure

    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separator=separator)

    texts = text_splitter.create_documents(documents)
    # Select embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    # Create a vectorstore from documents
    db = Chroma.from_documents(texts, embeddings)
    
    # Assuming vector_store is already initialized and populated as in the original code
    # And assuming you have a query
    query = query_text
    vectorstore = db

    
    #TODO: increase this at the end
    # Configuration for the retriever component of a RAG setup
    retriever_config = {
        "k": 4  # Adjust this value to increase or decrease the number of sources retrieved
    }

    # Initialize the retriever with the specified configuration
    retriever = vectorstore.as_retriever(search_kwargs=retriever_config)
    prompt = hub.pull("rlm/rag-prompt")
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)


    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | prompt
        | llm
        | StrOutputParser()
    )

    rag_chain_with_source = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)

    # Retrieve and generate using the relevant snippets from the document
    search_result_with_sources = rag_chain_with_source.invoke(query_text)
    print("___result_with_sources: ", search_result_with_sources)
    return search_result_with_sources

def safe_list_get(l, idx, default):
  try:
    return l[idx]
  except IndexError:
    return default
    
def get_references_text(file_name, response):
  references = '\n\n'.join(f"[{idx+1}] ({file_name}) {safe_list_get(response.get('context', []), idx, {}).page_content}" for idx in range(len(response.get('context', []))))
  return references

# Load environment variables from .env file
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

# Page title
st.set_page_config(page_title='Ontario Home Compliance AI Search')
st.title('Ontario Home Compliance AI Search')

# Add multi-select widget for selecting document
# documentation: st.multiselect(label, options, default=None, format_func=special_internal_function, key=None, help=None, on_change=None, args=None, kwargs=None, *, max_selections=None, placeholder="Choose an option", disabled=False, label_visibility="visible")
docs_selected = st.multiselect(
    'Pick the documents that you like to ask your questions to:',
    ['Ontario New Home Warranties Plan Act.txt', 'New Home Construction Licensing Act.txt', 'Application for Licences - Ontario Regulation 631.txt'],
    ['New Home Construction Licensing Act.txt'])
# st.write('You selected:', docs_selected)
#NOTE: currently the options are not used, below suduocode is the plan for doing this: 
# docs_selected = ['doc1.txt', 'doc2.txt']
# GPT_combined_answer_prompt = "here is the question we have:" +"below are list of relavant references paragraphs you can use to answer the questions, please using thse paragraphs to create an answer to the give question, and do not mix the content between different documents. if there are more than one documents that answers this question, please show the answer distinctively:"
# for doc_name in docs_selected:
#     response = generate_response(doc_name, openai_api_key, query_text)
#     GPT_combined_answer_prompt += "Document name: " + doc_name + "\n relavant paragraphs find:" + get_references_text_list(response)

# final_answer = openai.generate_response(GPT_combined_answer_prompt)

# File upload
file_name = st.file_uploader('Our upload your own document', type='pdf', accept_multiple_files = True)
# file_name = "New Home Construction Licensing Act.txt" #TODO: remove this at the end

# Query text
#NOTE: add this (disabled=not file_name) into the st.text_input() to stop flow from continuing
query_text = st.text_input('Enter your question:', placeholder='Please provide a short summary.') + "Please point out the section number and the specific actionable items that are relevant to the question. specifically information to do with the days"


# Form input and query
result = []
with st.form('myform', clear_on_submit=True):
    submitted = st.form_submit_button('Submit', disabled=not(query_text))
    if submitted and openai_api_key.startswith('sk-'):
        with st.spinner('Calculating...'):
            response = generate_response(file_name, openai_api_key, query_text)
            result.append(response)

if len(result):
    st.info(response.get('answer',""))
    st.info('\n\nReferences:\n' + get_references_text(file_name, response))

