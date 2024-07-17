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

import pandas as pd

def generate_response(company_info, openai_api_key, query_text):
    # Load document if file is uploaded
    # if company_info is not None:
    # Path to the local document file
    
    # current_directory = os.getcwd() 

    # Path to the local document file in the same folder, if file is locally stored
    # local_document_path = os.path.join(current_directory , 'docs/' + company_info)

    # Read the contents of the local document
    text = ''
    for file in company_info:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text += page.extract_text()

    # Assign the local document content to the 'documents' list
    documents = [text]

    # Split documents into chunks
    # Initialize the CharacterTextSplitter with best practices
    # Optimal chunk size and overlap for context preservation
    chunk_size = 300  # Adjust based on your document and model's token limit 
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
    
def get_references_text(response):
    reference_list = [safe_list_get(response.get('context', []), idx, {}).page_content for idx in range(len(response.get('context', [])))]

    reference_list = list(dict.fromkeys(reference_list))
    
    references_text = '\n\n'.join(f"[{idx+1}] {reference_list[idx]}" for idx in range(len(reference_list)))

    return references_text

# Load environment variables from .env file
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

# Page title
st.set_page_config(page_title='Form Autofill')
st.title('Form Autofill')

# Add multi-select widget for selecting document
# documentation: st.multiselect(label, options, default=None, format_func=special_internal_function, key=None, help=None, on_change=None, args=None, kwargs=None, *, max_selections=None, placeholder="Choose an option", disabled=False, label_visibility="visible")
# docs_selected = st.multiselect(
#     'Pick the documents that you like to ask your questions to:',
#     ['Ontario New Home Warranties Plan Act.txt', 'New Home Construction Licensing Act.txt', 'Application for Licences - Ontario Regulation 631.txt'],
#     ['New Home Construction Licensing Act.txt'])
# st.write('You selected:', docs_selected)
#NOTE: currently the options are not used, below suduocode is the plan for doing this: 
# docs_selected = ['doc1.txt', 'doc2.txt']
# GPT_combined_answer_prompt = "here is the question we have:" +"below are list of relavant references paragraphs you can use to answer the questions, please using thse paragraphs to create an answer to the give question, and do not mix the content between different documents. if there are more than one documents that answers this question, please show the answer distinctively:"
# for doc_name in docs_selected:
#     response = generate_response(doc_name, openai_api_key, query_text)
#     GPT_combined_answer_prompt += "Document name: " + doc_name + "\n relavant paragraphs find:" + get_references_text_list(response)

# final_answer = openai.generate_response(GPT_combined_answer_prompt)

######## UI for User Input Data ########
# Upload Instruction Document
instruction_document = st.file_uploader('Upload Instruction Document', type='pdf', accept_multiple_files=True, key='instruction_document_uploader')
# Upload Form to fill
form_document = st.file_uploader('Upload Form Document', type='csv', key='form_document_uploader')

# Add a file input for the company info
company_info = st.file_uploader('Upload Company Information Documents', type='pdf', accept_multiple_files=True, key='company_info_document_uploader')

######## UI for table display ########

# Create an empty placeholder for table
placeholder = st.empty()

# Assuming form_document is the file uploaded via st.file_uploader
if company_info is not None:
    if form_document is not None:
    # Read the uploaded CSV file into a DataFrame
        df = pd.read_csv(form_document)
        df['Answers'] = ''  
        df['References'] = ''
        # Update the placeholder with the new DataFrame
        placeholder.dataframe(df)
        df['Human Feedbacks'] = ''
    

######## Logic for searching dataframe ########

# Create a button in the sidebar
data_search_button = st.button('Click to Auto-fill Form with Company Info')

if data_search_button and 'df' in locals() and form_document is not None and company_info is not None:
    for index, row in df.iterrows():
        # Use the generate_response function to search for text in company_info within each row of the DataFrame
        response = generate_response(company_info, openai_api_key, row['Questions'])
        # Update the 'Answers' column
        df.at[index, 'Answers'] = response.get('answer',"")
        df.at[index, 'References'] = get_references_text(response)
        # Update the placeholder with the new DataFrame
        placeholder.dataframe(df)

# Button to regenerate the CSV after incorporating human feedbacks
regenerate_button = st.button('Regenerate CSV with Feedback')
if regenerate_button and 'df' in locals():
    # Assuming 'Human Feedbacks' column is manually updated in the UI before pressing the button
    st.write("Sorry this feature is not yet implimented")


######## Logic for download final filled form ########
# Add a download button for the DataFrame
if 'df' in locals():
    # Convert DataFrame to CSV string
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='edited_data.csv',
        mime='text/csv',
    )


# Form input and query
# result = []
# with st.form('myform', clear_on_submit=True):
#     submitted = st.form_submit_button('Submit', disabled=not(query_text))
#     if submitted and openai_api_key.startswith('sk-'):
#         with st.spinner('Calculating...'):
#             response = generate_response(company_info, openai_api_key, query_text)
#             result.append(response)

# if len(result):
#     st.info(response.get('answer',""))
#     st.info('\n\nReferences:\n' + get_references_text(instruction_document, response))
