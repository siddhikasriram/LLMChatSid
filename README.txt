# Enterprise Question Answering System - LLMChat

This project demonstrates a prototype of a question-answering system designed for enterprises, integrating data from various sources to provide insights from unstructured and semi-structured data. The system showcases the potential of building data pipelines to collect, process, and analyze diverse data formats, extracting meaningful information efficiently.

## Scripting Language
- **Python 3.9**

## Computation Details
- **CPU**

## Project Files
- **requirements.txt**: Contains all the packages needed to run the program.
- **final.py**: Streamlit application code for the chatbot interface.
- **chunkingDataEmbeddings.ipynb**: Contains the code for building data pipelines to extract data from different file formats, create word embeddings, and implement Retrieval-Augmented Generation (RAG).
- **sfdc-corpus**: Directory containing `.txt` files.
- **data**: Directory containing `.pdf` files.
- **gdrive-sampledata**: Directory containing a `.csv` file.
- **csv_files**: Directory containing multiple `.csv` files.
- **plan.jpg**: Snippet of the high-level architecture.

## Installation and Setup

### 1. Install the Required Packages
Install the required packages by running the following command:
```sh
pip install -r requirements.txt
```

### 2. Modify Path Configurations
- Update the paths for the data directories and the Chroma DB location in the `final.py` file to match your local setup.

### 3. API Key Configuration
- Before running the application, ensure you have the API key for **ChatGPT 3.5 Turbo**.

### 4. Run the Streamlit Application
To launch the question-answering system interface, run:
```sh
streamlit run final.py
```

## Explanation of `chunkingDataEmbeddings.ipynb`
- The notebook demonstrates how data pipelines are built to collect and process data from different sources.
- The collected data is transformed into vector embeddings, which are stored and used for building a **Retrieval-Augmented Generation (RAG)** model.
- For experimental purposes, only a single file (one `.pdf` file) is processed to showcase the system’s ability to handle diverse data formats, owing to computational constraints.

