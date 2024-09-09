import os
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
import openai
from langchain.text_splitter import TokenTextSplitter
from langchain.document_loaders import PyPDFLoader
import gspread
import pandas as pd
from PyPDF2 import PdfFileReader
from langchain.schema import Document
from InstructorEmbedding import INSTRUCTOR
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import Chroma

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from openai import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

import streamlit as st

#All the data after scraping is stored here

combined_data=[]

################## Change path ###################
path = "data/CHEF/10001727.pdf"

loader = PyPDFLoader(path)
combined_data.extend(loader.load())

# Split
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 5,
    separators=["\n\n", "\n", "(?<=\. )", " ", ""]
)

docs = text_splitter.split_documents(combined_data)

embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl", model_kwargs={"device": "cpu"},)

################## Change path ###################
persist_directory = '/Users/siddhikasriram/Documents/LLMChat/Chroma'

vectordb = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory=persist_directory
)

print(vectordb._collection.count())

question = "What skills are in the resume"

sim_docs = vectordb.similarity_search(question,k=3)

len(sim_docs)

vectordb.persist()

sim_docs_mmr = vectordb.max_marginal_relevance_search(question,k=2, fetch_k=3)
len(sim_docs_mmr)

# # Prompt the user to enter the OpenAI API key
api_key = input("Please enter your OpenAI API key: ")

# Set the environment variable with the provided API key
os.environ['OPENAI_API_KEY'] = api_key

print("API key has been set successfully.")

openai.api_key  = os.environ['OPENAI_API_KEY']

llm = ChatOpenAI(model_name='gpt-3.5-turbo')

retriever = vectordb.as_retriever(search_kwargs={"k": 3})

qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

st.title('LLMChat')

user_input = st.text_input("Enter a query: ")
if user_input:
    query = f"###Prompt {user_input}"
    try:
        llm_response = qa(query)
        st.write(llm_response["result"])
    except Exception as err:
        print('Exception occurred. Please try again', str(err))


