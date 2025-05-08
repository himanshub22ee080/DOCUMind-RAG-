import streamlit as st
import os
from dotenv import load_dotenv
import shutil
import traceback # For detailed error printing

# LangChain components
from langchain_community.document_loaders import PyPDFLoader # Using PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores.utils import filter_complex_metadata

# Import chromadb client directly for robust loading check
import chromadb

# --- CHOOSE YOUR MODELS ---
# For this version, we are hardcoding to Google AI with PyPDFLoader
USE_GOOGLE_AI = True
USE_OPENAI = False
USE_LOCAL_MODELS = False

if USE_GOOGLE_AI:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from langchain_google_genai import ChatGoogleGenerativeAI
# Other model imports would go here if flags were different

# --- Configuration ---
VECTOR_STORE_DIR = "pypdf_google_ai_vector_store" # Directory for this setup

# --- Core Functions ---

def clear_vector_store():
    """Clears the existing vector store directory."""
    if os.path.exists(VECTOR_STORE_DIR): # Use the global VECTOR_STORE_DIR
        try:
            shutil.rmtree(VECTOR_STORE_DIR)
            st.sidebar.info(f"Cleared vector store: '{VECTOR_STORE_DIR}'")
            # Reset relevant session state if store is cleared manually or due to error
            st.session_state.vector_db = None
            st.session_state.qa_chain = None
            # Keep processed_files_info? Maybe reset it too? Depends on desired UX.
            # st.session_state.processed_files_info = None
        except Exception as e:
            st.sidebar.error(f"Error clearing vector store: {e}")

def load_and_process_pdfs(pdf_files, chunk_size=1500, chunk_overlap=300):
    all_docs = []
    temp_upload_dir = "temp_pdf_uploads"
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
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    if not all_docs:
        st.warning("No documents could be loaded/processed from the uploaded PDF(s).")
        return None

    # Filter metadata
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

    # Apply RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len, is_separator_regex=False,
    )
    st.info(f"Splitting {len(all_docs)} pages/sections into smaller chunks...")
    split_docs = text_splitter.split_documents(all_docs)
    st.info(f"Total chunks created: {len(split_docs)}")

    if not split_docs:
        st.warning("No chunks were created after splitting. Check PDF content and splitter settings.")
        return None

    embeddings_model = None
    if USE_GOOGLE_AI:
        st.info("Using Google AI Embeddings (models/embedding-001)...")
        try:
            embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            st.info("Google AI Embeddings initialized.")
        except Exception as e:
            st.error(f"Error initializing Google AI Embeddings: {e}.")
            return None
    # Add other embedding model logic here

    if embeddings_model is None: 
        st.error("Embedding model could not be initialized.")
        return None
        
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True) # Ensure directory exists
    st.info(f"Creating new vector store embeddings for {len(split_docs)} chunks (using Google AI)...")
    try:
        # Use from_documents to create a NEW store based on the processed docs
        vector_db = Chroma.from_documents(
            documents=split_docs, 
            embedding=embeddings_model, 
            persist_directory=VECTOR_STORE_DIR,
            collection_name="langchain" # Use default collection name explicitly if needed
        )
        st.info("New vector store created successfully.")
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        print(f"ChromaDB error during creation:")
        print(traceback.format_exc())
        return None
    return vector_db

def get_conversational_qa_chain(vector_db):
    llm = None
    if USE_GOOGLE_AI:
        st.info("Initializing Google AI LLM (gemini-1.5-flash-latest)...")
        try:
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3, convert_system_message_to_human=True)
            st.info("Google AI LLM initialized.")
        except Exception as e:
            st.error(f"Error initializing Google AI LLM: {e}.")
            return None
    # Add other LLM logic here

    if llm is None: 
        st.error("LLM could not be initialized.")
        return None

    retriever = vector_db.as_retriever(search_kwargs={"k": 4}) 
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True, output_key='answer'
    )

    prompt_template_str = """Use the following pieces of context to answer the question at the end.
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

    conversational_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        output_key='answer',
        combine_docs_chain_kwargs={"prompt": QA_PROMPT} 
    )
    return conversational_chain

# --- Streamlit UI ---
def main():
    load_dotenv()
    st.set_page_config(page_title="DocuMind Q&A (PyPDF + Google AI)", layout="wide")
    
    # --- Optional: Clear vector store on first load of a session ---
    # Uncomment the block below if you PREFER to always start fresh on new sessions
    # on ephemeral storage like Streamlit Cloud, ignoring potential persistence.
    # if "initialized" not in st.session_state:
    #     print(f"Attempting to clear vector store '{VECTOR_STORE_DIR}' on app session start...")
    #     clear_vector_store()
    #     st.session_state.initialized = True 
    # --- End Optional Clear ---

    st.title("📄 DocuMind Q&A")
    model_name_display = "Google AI" 
    parser_name_display = "PyPDF"     
    st.caption(f"✨ Powered by {model_name_display} | Parser: {parser_name_display} | Conversational Mode")

    # Initialize session state
    if "vector_db" not in st.session_state: st.session_state.vector_db = None
    if "qa_chain" not in st.session_state: st.session_state.qa_chain = None
    if "processed_files_info" not in st.session_state: st.session_state.processed_files_info = None
    if "messages" not in st.session_state: st.session_state.messages = []

    with st.sidebar:
        st.header("📁 Document Setup")
        st.info(f"Current Mode: **{model_name_display}** with **{parser_name_display}**")
        
        uploaded_files = st.file_uploader(
            "Upload PDF files", type="pdf", accept_multiple_files=True, key="pdf_uploader_pypdf_v3"
        )
        force_reprocess = st.checkbox(
            f"Force re-process & clear {parser_name_display} vector store", key="reprocess_pypdf_v3", 
            help=f"This will delete and recreate the '{VECTOR_STORE_DIR}' directory."
        )

        if st.button(f"Process Uploaded PDF(s)", key="process_button_pypdf_v3", disabled=not uploaded_files):
            with st.spinner(f"Processing PDFs with {parser_name_display} & {model_name_display}..."):
                
                # --- Logic for Handling Vector Store ---
                load_existing = False
                # 1. Check if forced reprocess is selected
                if force_reprocess:
                    clear_vector_store() # Clears disk and resets session state vector_db/qa_chain
                    st.session_state.messages = [] # Clear messages on forced reprocess
                else:
                    # 2. Check if a store directory exists and session info matches
                    if os.path.exists(VECTOR_STORE_DIR) and st.session_state.processed_files_info:
                        if st.session_state.processed_files_info.get("parser", "").lower() == parser_name_display.lower() and \
                           st.session_state.processed_files_info.get("model_type") == model_name_display:
                            load_existing = True # Potentially load
                        else:
                            # Mismatch in config, clear the old store
                            st.warning("Configuration mismatch detected. Clearing incompatible vector store.")
                            clear_vector_store()
                            st.session_state.messages = []
                    elif os.path.exists(VECTOR_STORE_DIR) and not st.session_state.processed_files_info:
                        # Store exists, but we have no info about it. Assume incompatible/stale.
                         st.warning("Found existing vector store with unknown configuration. Clearing it.")
                         clear_vector_store()
                         st.session_state.messages = []


                # 3. Attempt to load if conditions met
                if load_existing:
                    st.info(f"Attempting to load existing vector store '{VECTOR_STORE_DIR}'...")
                    try:
                        # More robust check: use chromadb client directly
                        client = chromadb.PersistentClient(path=VECTOR_STORE_DIR)
                        client.get_collection(name="langchain") # Check if default collection exists
                        
                        # If successful, load using LangChain wrapper
                        embeddings_model_loader = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                        st.session_state.vector_db = Chroma(
                            persist_directory=VECTOR_STORE_DIR, 
                            embedding_function=embeddings_model_loader,
                            collection_name="langchain" # Match default
                        )
                        st.info("Loaded existing vector store.")
                        # We already have processed_files_info, don't reset messages
                    except Exception as e:
                        # --- If loading fails, clear and prepare for reprocessing ---
                        st.warning(f"Could not load existing vector store (may be corrupt or incompatible): {e}. Clearing and re-processing.")
                        clear_vector_store() # Deletes dir, resets session state vector_db/qa_chain
                        st.session_state.messages = [] # Clear messages if store was corrupt
                        load_existing = False # Ensure we proceed to process
                        # --- End clearing ---

                # 4. Process PDFs if not loaded or if vector_db is None after failed load attempt
                if not load_existing or st.session_state.vector_db is None:
                    # Ensure store is clear if we are about to create a new one
                    if not force_reprocess and os.path.exists(VECTOR_STORE_DIR):
                         # If loading wasn't attempted or failed, ensure dir is clear before creating new
                         clear_vector_store() 
                         st.session_state.messages = []

                    # Process PDFs and create a new vector store
                    st.session_state.vector_db = load_and_process_pdfs(uploaded_files)
                    st.session_state.messages = [] # Clear messages when new docs processed


                # 5. Create QA chain if vector_db is available
                if st.session_state.vector_db:
                    st.session_state.qa_chain = get_conversational_qa_chain(st.session_state.vector_db)
                    # Update processed files info only if we actually processed or successfully loaded
                    st.session_state.processed_files_info = {
                        "names": [f.name for f in uploaded_files], 
                        "model_type": model_name_display,
                        "parser": parser_name_display 
                    }
                    st.success(f"PDFs ready! Ready to chat.")
                else:
                    # Ensure chain is cleared if processing failed
                    st.session_state.qa_chain = None 
                    st.session_state.processed_files_info = None
                    st.error(f"PDF processing failed. Vector store could not be created/loaded.")
                # --- End Logic for Handling Vector Store ---

        if st.session_state.processed_files_info:
            parser_info = f" (Parser: {st.session_state.processed_files_info.get('parser', 'Unknown')})"
            st.success(f"Active Docs ({st.session_state.processed_files_info['model_type']}{parser_info}):")
            for name in st.session_state.processed_files_info["names"]: st.markdown(f"- `{name}`")
        
        if st.button("Clear Chat History", key="clear_chat_pypdf_v3"):
            st.session_state.messages = []
            st.rerun()

    st.header("💬 Chat with Your Documents")
    # (Chat display and input logic remains the same as the previous good version)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=True) 
            if "sources" in message and message["sources"]:
                with st.expander("Sources Used"):
                    for src in message["sources"]: st.markdown(f"- `{src}`")

    if st.session_state.qa_chain:
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
