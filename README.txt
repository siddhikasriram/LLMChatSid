Enterprise Question Answering System - LLMChat
This project demonstrates a prototype of a question answering system designed for enterprises, integrating data from various sources to provide insights from unstructured and semi-structured data. The system showcases the potential of integrating diverse data formats and processing them to extract meaningful information.
Scripting Language
* Python 3.9
Computation Details
* CPU
Project Files
* requirements.txt: Contains all the packages needed to run the program.
* final.py: Streamlit application code for the chatbot interface.
* chunkingDataEmbeddings.ipynb: Contains the code for data extraction from different file formats, creating word embeddings, and Retrieval-Augmented Generation (RAG).
* sfdc-corpus: Directory containing .txt files.
* data: Directory containing .pdf files.
* gdrive-sampledata: Directory containing a .csv file.
* csv_files: Directory containing multiple .csv files.
* plan.jpg: Snippet of the high-level architecture.
Installation and Setup
1. Install the Required Packages
Install the required packages by running the following command:
pip install -r requirements.txt
   2. Modify Path Configurations
   * Update the paths for the data directories and the Chroma DB location in the final.py file to match your local setup.


   3. API Key Configuration
   * Before running the application, ensure to have the key for ChatGPT 3.5 turbo.


   4. Run the Streamlit application.
To launch the question-answering system interface, run:
streamlit run final.py
Explanation of chunkingDataEmbeddings.ipynb
   * The notebook demonstrates how data is read from different sources and processed to create vector embeddings.
   * The vector embeddings are stored and used for building a Retrieval-Augmented Generation (RAG) model.
   * For experimental purposes, only a single file (one .pdf file) is processed to showcase the system’s ability to handle diverse data formats, owing to computational constraints.