import streamlit as st
import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain


# PAGE SETUP
st.set_page_config(page_title="My Second Brain", layout="wide")
st.title("🧠 Chat with your Notes")

# SETTINGS
with st.sidebar:
    st.header("Settings")
    folder_path = st.text_input("Path to your notes folder:", value="./my_notes")

    if st.button("Build/Update Knowledge Base"):
        with st.spinner("Reading notes and building vectors..."):
            try:
                # 1. LOAD DOCUMENTS
                if not os.path.exists(folder_path):
                    st.error("The specified folder does not exist!")
                else:
                    loader = DirectoryLoader(folder_path, glob="**/*.md", loader_cls=TextLoader)
                    documents = loader.load()

                    if not documents:
                        st.error("No markdown files found in that folder.")
                    else:
                        # 2. SPLIT TEXT
                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                        docs = text_splitter.split_documents(documents)

                        # 3. EMBED & STORE
                        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                        vector_store = FAISS.from_documents(docs, embeddings)
                        vector_store.save_local("faiss_index")

                        st.success(f"Success! Processed {len(docs)} chunks from your notes.")

            except Exception as e:
                st.error(f"Error details: {e}")

# --- MAIN CHAT INTERFACE ---

# Initialize the LLM
llm = ChatOllama(model="llama3")

# Check if the vector store exists before trying to chat
if os.path.exists("faiss_index"):
    # Load the vector store
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # THE NEW CHAIN IMPLEMENTATION

    # 1. Define the Prompt
    # BEHAVE CONSTRAINT
    # This tells the LLM exactly how to behave.
    prompt = ChatPromptTemplate.from_template("""
    Answer the following question based ONLY on the provided context. 
    If the answer is not in the context, say "I don't see that in your notes."

    <context>
    {context}
    </context>

    Question: {input}
    """)

    # 2. Create the "Document Chain"
    # This chain handles sending the documents + prompt to the LLM
    document_chain = create_stuff_documents_chain(llm, prompt)

    # 3. Create the "Retrieval Chain"
    # This chain combines the retriever (fetching docs) with the document chain (answering)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # CHAT UI

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if input_text := st.chat_input("Ask something about your notes..."):

        with st.chat_message("user"):
            st.markdown(input_text)
        st.session_state.messages.append({"role": "user", "content": input_text})

        with st.chat_message("assistant"):
            with st.spinner("Analyzing  your notes..."):
                # Run the chain
                # Note: We use "input" because that's what we defined in the prompt template above
                response = retrieval_chain.invoke({"input": input_text})

                answer = response["answer"]

                # Extract sources (unique filenames)
                # In the new chain, source docs are in response["context"]
                sources = {doc.metadata['source'] for doc in response["context"]}

                st.markdown(answer)
                if sources:
                    st.caption(f"Sources: {', '.join(sources)}")

        st.session_state.messages.append({"role": "assistant", "content": answer})

else:
    st.info("👈 Please enter your notes folder path and click 'Build Knowledge Base' to start.")