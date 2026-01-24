# 🧠 AI Chatbot for Obsidian Notes or any .md file

I have hundreds of Markdown notes across my laptop from my Obsidian vault, so I wanted to build a chatbot that has as knowledge base the notes, so that I can make resumes of them.
It uses a local LLM to understand the meaning of my questions, retrieves the relevant specific notes, and generates an answer based only on my data.
No data leaves my laptop. It runs 100% locally.

## How It Works (The RAG Pipeline)

The system follows a standard RAG architecture that I implemented to handle unstructured Markdown text:

- Ingestion: The app scans my notes folder and loads all .md files.
- Chunking: It uses a RecursiveCharacterTextSplitter to break text into manageable chunks (500 chars) while preserving context overlap.
- Embedding: Each chunk is converted into a vector using the MiniLM model.
- Storage: Vectors are stored in a local FAISS index.
- Retrieval & Generation: When I ask a question, the system finds the top 3 most similar chunks and feeds them into Llama 3 with a strict prompt to answer only using that context.

## How to Run It

1. Clone the repo
```python
git clone https://github.com/Fmayo444/Obsidian_notes_chatbot.git
cd Obsidian_notes_chatbot
```

2. Set up the environment Make sure you are using Python 3.11 (important for FAISS and Langchain compatibility!).

```python
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

3. Install Ollama. You need the Ollama desktop app to run the LLM backend.

    Download it from ollama.com.

    Pull the model (I use Llama 3):
   
```python
    ollama run llama3
```

4. Run the App

```python
streamlit run main.py
```
 



    

    
  
  
