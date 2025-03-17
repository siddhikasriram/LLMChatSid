# Enterprise Question Answering System - LLMChat

This repository contains a prototype of a question-answering system designed for enterprises, integrating data from various sources to provide insights from unstructured and semi-structured data. The system highlights the potential of building data pipelines to collect, process, and analyze diverse data formats efficiently.

## Tools and Technologies

- **Scripting Language:** Python 3.9  
- **Computational Platform:** CPU  

## File Structure

- `requirements.txt`: Contains all the packages needed to run the program.  
- `final.py`: Streamlit application code for the chatbot interface.  
- `chunkingDataEmbeddings.ipynb`: Jupyter notebook for building data pipelines to extract data from different file formats, create word embeddings, and implement Retrieval-Augmented Generation (RAG).  
- `sfdc-corpus/`: Directory containing `.txt` files.  
- `data/`: Directory containing `.pdf` files.  
- `gdrive-sampledata/`: Directory containing a `.csv` file.  
- `csv_files/`: Directory containing multiple `.csv` files.  
- `plan.jpg`: Snippet of the high-level architecture.  

## Installation and Setup

1. **Installation:** Install the required packages by running:  
   ```sh
   pip install -r requirements.txt
   ```  
2. **Modify Path Configurations:** Update the paths for the data directories and the Chroma DB location in `final.py` to match your local setup.  
3. **API Key Configuration:** Ensure you have the API key for **ChatGPT 3.5 Turbo** before running the application.  
4. **Execution:** Launch the question-answering system interface by running:  
   ```sh
   streamlit run final.py
   ```

## Data Pipeline and Processing

- The notebook `chunkingDataEmbeddings.ipynb` demonstrates how data pipelines are built to collect and process data from different sources.  
- The collected data is transformed into vector embeddings, which are stored and used for building a **Retrieval-Augmented Generation (RAG)** model.  
- For experimental purposes, only a single file (one `.pdf` file) is processed to showcase the system’s ability to handle diverse data formats, owing to computational constraints.  

## Additional Information

For detailed documentation and explanations, please refer to the respective code files in this repository.  

## Author and Contact

- **Author:** Siddhika Sriram  
- **Contact:** [Email](mailto:siddhikasriram.ss@gmail.com), [LinkedIn](https://www.linkedin.com/in/sidsriram/)  
