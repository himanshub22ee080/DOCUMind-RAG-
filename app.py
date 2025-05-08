# Override system sqlite3 with pysqlite3-binary (Still needed as ChromaDB uses sqlite3 even in-memory)
try:
    print("Attempting to override sqlite3 with pysqlite3...")
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    print("Successfully replaced sqlite3 with pysqlite3.")
except ImportError:
    print("pysqlite3 not found, using standard sqlite3.")
    pass
except KeyError:
     print("pysqlite3 already loaded or override failed.")
     pass

# --- Now your regular imports ---
import streamlit as st
import os
from dotenv import load_dotenv
# import shutil # No longer needed for directory operations
import traceback # For detailed error printing

# LangChain components
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores.utils import filter_complex_metadata

# Import chromadb client not strictly needed for in-memory, but doesn't hurt
# import chromadb 

# --- CHOOSE YOUR MODELS ---
USE_GOOGLE_AI = True
USE_OPENAI = False
USE_LOCAL_MODELS = False

if USE_GOOGLE_AI:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from langchain_google_genai import ChatGoogleGenerativeAI
# Other model imports if needed

# --- Configuration ---
# VECTOR_STORE_DIR = "pypdf_google_ai_vector_store" # --- CHANGE --- No longer needed

# --- Core Functions ---

# def clear_vector_store(): # --- CHANGE --- No longer needed for in-memory
#     pass 

def load_and_process_pdfs(pdf_files, chunk_size=1500, chunk_overlap=300):
    all_docs = []
    temp_upload_dir = "temp_pdf_uploads"
    # Ensure temp dir exists, but we won't persist the vector store itself
    os.makedirs(temp_upload_dir, exist_ok=True) 

    for pdf_file in pdf_files:
        temp_file_path = os.path.join(temp_upload_dir, pdf_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(pdf_file.getbuffer())
        
        st.info(f"Processing '{pdf_file.name}' with PyPDFLoader...")
        loader = PyPDFLoader(temp_file_path)
        try:
            loaded_docs_for_file = loader.load()
            for doc in loaded_docs_for_file:
                doc.metadata["source"] = pdf_file.name 
            all_docs.extend(loaded_docs_for_file)
            st.success(f"Successfully processed '{pdf_file.name}' with PyPDFLoader.")
        except Exception as e:
            st.error(f"Error loading/processing PDF '{pdf_file.name}' with PyPDFLoader: {e}. Skipping.")
            print(f"PyPDFLoader error for {pdf_file.name}:")
            print(traceback.format_exc())
        finally:
            # --- CHANGE --- Clean up temp PDF file
            if os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except Exception as e_rem:
                     print(f"Error removing temp file {temp_file_path}: {e_rem}")
            # --- END CHANGE ---

    if not all_docs:
        st.warning("No documents could be loaded/processed from the uploaded PDF(s).")
        return None

    st.info(f"Loaded {len(all_docs)} pages/sections. Filtering complex metadata...")
    try:
        all_docs = filter_complex_metadata(all_docs)
        st.success("Complex metadata filtered.")
    except Exception as e:
        st.error(f"Error during metadata filtering: {e}")
        return None

    if not all_docs:
        st.warning("No documents available after metadata filtering.")
        return None

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len, is_separator_regex=False,
    )
    st.info(f"Splitting {len(all_docs)} pages/sections into smaller chunks...")
    split_docs = text_splitter.split_documents(all_docs)
    stOkay, let's implement **Option 1: Using ChromaDB In-Memory**.

This means ChromaDB will not attempt.info(f"Total chunks created: {len(split_docs)}")

    if not split_docs:
        st.warning("No chunks were created after splitting. Check PDF content and splitter settings.")
        return None

    embeddings_model = None
    if USE_GOOGLE_AI:
        st.info("Using Google AI Embeddings (models/embedding-001)...")
        try:
            embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            st.info("Google AI Embeddings initialized to save anything to the disk inside your Streamlit Cloud container, avoiding all the filesystem and persistence-related errors. The.")
        except Exception as e:
            st.error(f"Error initializing Google AI Embeddings: {e}.")
            return None
    
    if embeddings_model is None: 
        st.error("Embedding model could not be initialized.")
        return None
        
    # --- CHANGE --- Create In-Memory Vector Store trade-off is that the vector store exists only for the duration of that specific app instance run – when the app sleeps or restarts, the data is gone, and PDFs must be reprocessed.

**Step-by-Step Changes:**

**1. Modify `requirements.txt` (Optional but Recommended Cleanup):**

*   Since we are no longer persisting to disk, the `pysqlite3-binary` package is not strictly needed anymore (as we aren't fighting the system's `sqlite3` version for persistence). You can remove it to slightly simplify the environment. The `protobuf` pin might still be needed for other Google libraries.
*   Open `requirements.txt`:
    ```
    streamlit

    # os.makedirs(VECTOR_STORE_DIR, exist_ok=True) # No directory needed
    st.info(f"Creating IN-MEMORY vector store embeddings for {len(split_docs)} chunks...")
    try:
        # When no persist_directory is provided, Chroma runs in memory
        vector_db = Chroma.from_documents(
            documents=split_docs, 
            embedding=embeddings_model,
            # persist    langchain
    langchain-community
    langchain-google-genai
    pypdf
    # pysqlite3-binary  # <-- REMOVE OR COMMENT OUT
    chromadb>=0.4.15,<0.5.0 # Keep pinned version for stability
    python-dotenv
    tiktoken
    protobuf_directory=VECTOR_STORE_DIR, # REMOVED
            collection_name="langchain_in_memory" # Optional name
        )
        st.info("In-memory vector store created successfully.")
    except Exception as e:
        st.error(f"Error creating in-memory vector store: {e}")
        print(f"ChromaDB in-memory error during creation:")
        print(traceback.format_exc())
        return None
==3.20.3
    setuptools
    numpy<2.0.0
    ```
*   Save `requirements.txt`.
*   Run `pip install -r requirements.txt` locally to update your environment (or just ensure `pysqlite3-binary` is removed if you prefer `pip uninstall pysqlite3-    # --- END CHANGE ---
    return vector_db

def get_conversational_qa_chain(vector_db):
    # --- This function remains exactly the same as before ---
    # It takes the vector_db object (which is now in-memory) and creates the chain
    llm = None
    if USE_GOOGLE_AI:
binary`).

**2. Modify `app.py`:**

*   **Remove SQLite Override:** Delete the `try...except` block at the very top of the file that overrides `sqlite3` with `pysqlite3`.
*   **Remove `VECTOR_STORE_DIR`:** Delete the global configuration variable `VECTOR_STORE_DIR`.
*   **Remove `clear_vector_store` Function:** Delete the entire `clear_vector_store`        st.info("Initializing Google AI LLM (gemini-1.5-flash-latest)...")
        try:
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3, convert_system_message_to_human=True)
            st.info("Google AI LLM initialized.")
        except Exception as e:
            st.error(f"Error initializing Google AI LLM: {e}.")
            return None
    if llm is None: 
 function definition.
*   **Update `load_and_process_pdfs`:** Remove the `persist_directory` argument when calling `Chroma.from_documents`. Remove checks/creation related to `VECTOR_STORE_DIR`.
*   **Update `main` Function:** Remove all logic related to `VECTOR_STORE_DIR`, `clear_vector_        st.error("LLM could not be initialized.")
        return None

    retriever = vector_db.as_retriever(search_kwargs={"k": 4}) 
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True, output_key='answer'
    )
    prompt_template_str = """Use the following pieces of context to answer the question at the end.
    Your main goal is to provide a detailed, comprehensive, and to-the-point answer based SOLELY on the provided context.
store`, loading existing stores, and `force_reprocess` (as reprocessing is now implicit whenever the app starts fresh or the button is clicked). Simplify the button logic.

**Here is the entire modified `app.py`:**

```python
# No longer need the pysqlite3 override at the top

import streamlit as st
import os
from dotenv import load_dotenv
# No longer need shutil if not clearing directories
import traceback

# LangChain components
from    If the context includes specific examples, data points, or step-by-step instructions, please try to include them in your answer.
    Structure your answer clearly.
    If you don't know the answer from the context or if the context is insufficient, clearly state that you cannot answer based on the provided documents. Do not make up information.

    Context:
    {context}

    Chat History:
    {chat_history}

    Question: {question}
    Helpful Answer:"""
    QA_PROMPT = PromptTemplate(
        template=prompt_template_str, input_variables=["context", "chat_history", "question"]
    )
    conversational_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores.utils import filter_complex_metadata

# No longer need chromadb client import for loading check
# import chromadb

# --- CHOOSE YOUR MODELS ---
USE_GOOGLE_AI = True
USE_OPENAI = False
USE_LOCAL_MODELS = False

if USE_GOOGLE_AI:
    from langchain_google_genai import=True,
        output_key='answer',
        combine_docs_chain_kwargs={"prompt": QA_PROMPT} 
    )
    return conversational_chain
    # --- End of unchanged get_conversational_qa_chain ---

# --- Streamlit UI ---
def main():
    load_dotenv()
    st.set_page_config(page_title="DocuMind Q&A (In-Memory)", layout="wide")
    st.title("📄 DocuMind Q&A")

    model_name_display = "Google AI" 
    parser_name_display = "PyPDF"     
    storage_type = "In-Memory" # Indicate storage type
    st.caption(f"✨ Powered by {model_name_display} GoogleGenerativeAIEmbeddings
    from langchain_google_genai import ChatGoogleGenerativeAI
# Other model imports

# --- Configuration ---
# VECTOR_STORE_DIR = "pypdf_google_ai_vector_store" # REMOVED - No longer persisting

# --- Core Functions ---

# def clear_vector_store(): # REMOVED - No longer needed
#     pass

def load_and_process_pdfs(pdf_files, chunk_size=1500, chunk_overlap=300):
    # (PDF loading loop remains the same)
    all_docs = []
    temp_upload_dir = "temp | Parser: {parser_name_display} | Storage: {storage_type} | Conversational Mode")

    # Initialize session state 
    # No need for 'initialized' flag as we don't clear on startup anymore
    if "vector_db" not in st.session_state: st.session_state.vector_db = None
    if "qa_chain" not in st.session_state: st.session_state.qa_chain = None
    if "processed_files_info" not in st.session_state: st.session_state.processed_files_info = None
    if "messages" not in st.session_state: st.session_state.messages = []

    with st.sidebar:
        st.header("📁_pdf_uploads"
    os.makedirs(temp_upload_dir, exist_ok=True)
    for pdf_file in pdf_files:
        temp_file_path = os.path.join(temp_upload_dir, pdf_file.name)
        with open(temp_file_path, "wb") as f: f.write(pdf_file.getbuffer())
        loader = PyPDFLoader(temp_file_path)
        try:
            loaded_docs_for_file = loader.load()
            for doc in loaded_docs_for_file: doc.metadata["source"] = pdf_file.name 
            all_docs.extend(loaded_docs_for_file)
            st.success(f"Successfully processed '{pdf_file.name}' with PyPDFLoader.")
        except Exception as e:
            st.error Document Setup")
        st.info(f"Current Mode: **{model_name_display}** with **{parser_name_display}**")
        st.warning("Vector store is In-Memory: PDFs must be re-processed each session.") # Inform user
        
        uploaded_files = st.file_uploader(
            "Upload PDF(f"Error loading PDF '{pdf_file.name}': {e}. Skipping.")
            print(f"PyPDFLoader error for {pdf_file.name}:"); print(traceback.format_exc())
        finally:
            if os.path.exists(temp_file_path): os.remove(temp_file_path)
    if not all_docs:
        st.warning("No documents loaded."); return None

     files", type="pdf", accept_multiple_files=True, key="pdf_uploader_inmemory" 
        )
        
        # force_reprocess checkbox is removed as processing always happens from scratch now

        if st.button(f"Process Uploaded PDF(s)", key="process_button_inmemory", disabled=not uploaded_files):
            with st.spinner(f"Processing PDFs into memory..."):
                # --- Simplified# (Metadata filtering remains the same)
    st.info(f"Loaded {len(all_docs)} pages/sections. Filtering complex metadata...")
    try:
        all_docs = filter_complex_metadata(all_docs)
        st.success("Complex metadata filtered.")
    except Exception as e:
        st.error(f"Error filtering metadata: {e}"); return None
    if not all_docs:
        st.warning("No documents after filtering."); return None

    # (Text splitting remains the same)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function= Processing Logic for In-Memory ---
                # Always start fresh for session state when button is clicked
                st.session_state.vector_db = None # Clear potentially old in-memory db object
                st.session_state.qa_chain = None
                st.session_state.messages = [] 
                st.session_state.processed_files_info = None

                # Process PDFs and create a new IN-MEMORY vector store
                st.session_state.vector_db = load_and_process_pdfs(uploaded_files)
                
len, is_separator_regex=False,
    )
    st.info(f"Splitting {len(all_docs)} pages/sections into chunks...")
    split_docs = text_splitter.split_documents(all_docs)
    st.info(f"Total chunks created: {len(split_docs)}")
    if not split_docs:
        st.warning("No chunks created."); return None

    # (Embedding model initialization remains the same)
    embeddings_model = None
    if USE_GOOGLE_AI:
        st                # Create QA chain IF vector_db was successfully created
                if st.session_state.vector_db:
                    st.session_state.qa_chain = get_conversational_qa_chain(st.session_state.vector_db)
                    # Update processed files info
                    st.session_state.processed_files_info = {
                        "names": [f.name for f in uploaded_files], 
                        "model_type": model_name_display,
                        "parser": parser_name_display,
                        "storage.info("Using Google AI Embeddings (models/embedding-001)...")
        try:
            embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            st.info("Google AI Embeddings initialized.")
        except Exception as e: st.error(f"Error initializing Embeddings: {e}."); return None
    if embeddings_model is None: st.error("Embedding model not initialized.");": storage_type 
                    }
                    st.success(f"PDFs processed (in memory)! Ready to chat.")
                else:
                    # Ensure chain/info is cleared if processing failed
                    st.session_state.qa_chain = None 
                    st.session_state.processed_files_info = None
                    st.error(f"PDF processing failed. In-memory vector store could not be created.")
                # --- End Simplified Logic ---
        
        if st.session_state.processed_files_info:
            parser_info = f" (Parser: {st.session_state.processed_files_info.get('parser', 'Unknown')})"
            storage_info = f" (Storage: {st.session_state.processed_files_info.get('storage', 'Unknown')})"
            st.success(f"Active Docs ({st return None
        
    # --- Create IN-MEMORY Chroma vector store ---
    # os.makedirs(VECTOR_STORE_DIR, exist_ok=True) # REMOVED
    st.info(f"Creating IN-MEMORY vector store with {len(split_docs)} chunks...")
    try:
        vector_db = Chroma.from_documents(
            documents=split_docs, 
            embedding=embeddings_model
            # persist_directory=VECTOR_STORE_DIR, # REMOVED - This makes it in-memory
            # collection_name="langchain" # Default name is usually fine for in-memory
        )
        st.info("In-memory vector store created successfully.")
    except Exception as e:
        st.error(f"Error creating.session_state.processed_files_info['model_type']}{parser_info}{storage_info}):")
            for name in st.session_state.processed_files_info["names"]: st.markdown(f"- `{name}`")
        
        if st.button("Clear Chat History", key="clear_chat_inmemory"):
            st.session_state.messages = []
            st.rerun() # Rerun to clear the chat display

    st.header("💬 Chat with Your Documents")
    # (Chat display and input logic remains the same as the previous good version)
    for message in st.session_state.messages:
 in-memory vector store: {e}")
        print(f"ChromaDB in-memory error:"); print(traceback.format_exc())
        return None
    return vector_db
    # --- End In-Memory Change ---

# (get_conversational_qa_chain remains the same as the previous version with the custom prompt)
def get_conversational_qa_chain(vector_db):
    llm = None
    if USE_GOOGLE_AI:
        st.info("Initializing Google AI LLM (gemini-1.5-flash-latest)...")
        try:
            llm = ChatGoogleGenerativeAI(model="gem        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=True) 
            if "sources" in message and message["sources"]:
                with st.expander("Sources Used"):
                    for src in message["sources"]: st.markdown(f"- `{src}`")

ini-1.5-flash-latest", temperature=0.3, convert_system_message_to_human=True)
            st.info("Google AI LLM initialized.")
        except Exception as e: st.error(f"Error initializing LLM: {e}."); return None
    if llm is None: st.error("LLM not initialized."); return None

    retriever = vector_db.as_retriever(search_kwargs={"k": 4}) 
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True, output_key='answer'
    )
    prompt_template_str    if st.session_state.qa_chain:
        if prompt := st.chat_input("Ask a question about your documents..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.markdown(prompt)
            with st.chat_message("assistant"):
                msg_placeholder = st.empty()
                with st.spinner("Thinking with Gemini..."):
                    try:
                        response = st.session_state.qa_chain.invoke({"question": prompt})
                        answer = response.get("answer", "Sorry, I couldn't find an answer.")
                        sources = []
                        if "source_documents" in response and response["source_documents"]:
                            sources = list(set(os.path.basename(doc.metadata.get("source", "Unknown")) for doc in response["source_documents"]))
                        
                        msg_placeholder.markdown(answer, unsafe_allow_html=True)
                        
                        if sources:
                            with st.expander("Sources Used"):
 = """Use the following pieces of context to answer the question at the end.
    Your main goal is to provide a detailed, comprehensive, and to-the-point answer based SOLELY on the provided context.
    If the context includes specific examples, data points, or step-by-step instructions, please try to include them in your answer.
    Structure your answer clearly.
    If you don't know the answer from the context or if the context is insufficient, clearly state that you cannot answer based on the provided documents. Do not make up information.

    Context:
    {context}

    Chat History:
    {chat_history}

    Question: {question}
    Helpful Answer:"""
    QA_PROMPT = PromptTemplate(
        template=prompt_template_str, input_variables=["context", "chat_history", "question"]
    )
                                for src in sources: st.markdown(f"- `{src}`")
                        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
                    except Exception as e:
                        err_msg = f"An error occurred: {e}"
                        st.error(err_msg)
                        st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}", "sources": []})
                        print(f"ERROR in QA Chain: {traceback.format_exc()}")
                        msg_placeholder.markdown("Sorry, an error occurred. Check terminal logs.")
    else:
        st.info("Please upload and process PDF files using the sidebar to start chatting.")

# Corrected the entry point check
if __name__ == "__main__":
    main()
