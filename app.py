# No pysqlite3 override needed at the top anymore

import streamlit as st
import os
from dotenv import load_dotenv
# import shutil # No longer needed
import traceback # For detailed error printing

# LangChain components
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS # <-- Import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores.utils import filter_complex_metadata

# --- CHOOSE YOUR MODELS ---
# Hardcoding to Google AI for this version
USE_GOOGLE_AI = True
USE_OPENAI = False
USE_LOCAL_MODELS = False

if USE_GOOGLE_AI:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from langchain_google_genai import ChatGoogleGenerativeAI
# Other model imports if needed

# --- Configuration ---
# No VECTOR_STORE_DIR needed for in-memory FAISS

# --- Core Functions ---

# No clear_vector_store function needed

def load_and_process_pdfs(pdf_files, chunk_size=1500, chunk_overlap=300):
    """Loads PDFs, splits them, filters metadata, and creates an IN-MEMORY FAISS vector store."""
    all_docs = []
    temp_upload_dir = "temp_pdf_uploads"
    os.makedirs(temp_upload_dir, exist_ok=True)

    for pdf_file in pdf_files:
        temp_file_path = os.path.join(temp_upload_dir, pdf_file.name)
        try:
            with open(temp_file_path, "wb") as f:
                f.write(pdf_file.getbuffer())

            st.info(f"Processing '{pdf_file.name}' with PyPDFLoader...")
            loader = PyPDFLoader(temp_file_path)
            loaded_docs_for_file = loader.load()
            for doc in loaded_docs_for_file:
                doc.metadata["source"] = pdf_file.name
            all_docs.extend(loaded_docs_for_file)
            st.success(f"Successfully loaded text from '{pdf_file.name}'.")
        except Exception as e:
            st.error(f"Error loading/processing PDF '{pdf_file.name}': {e}. Skipping.")
            print(f"PyPDFLoader error for {pdf_file.name}:")
            print(traceback.format_exc())
        finally:
            # Clean up temp PDF file
            if os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except Exception as e_rem:
                     print(f"Error removing temp file {temp_file_path}: {e_rem}")

    if not all_docs:
        st.warning("No documents could be loaded/processed from the uploaded PDF(s).")
        return None

    # Filter metadata
    st.info(f"Loaded {len(all_docs)} pages/sections. Filtering complex metadata...")
    try:
        if all_docs: # Check if list is not empty before filtering
            all_docs = filter_complex_metadata(all_docs)
            st.success("Complex metadata filtered.")
    except Exception as e:
        st.error(f"Error during metadata filtering: {e}")
        return None

    if not all_docs: # Check again after filtering
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

    # Initialize Embedding Model
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

    # --- Create IN-MEMORY FAISS vector store ---
    st.info(f"Creating IN-MEMORY FAISS index with {len(split_docs)} chunks...")
    try:
        # FAISS.from_documents creates the index in memory directly
        vector_db = FAISS.from_documents(
            documents=split_docs,
            embedding=embeddings_model
        )
        st.info("In-memory FAISS index created successfully.")
    except Exception as e:
        st.error(f"Error creating in-memory FAISS index: {e}")
        print(f"FAISS error during creation:")
        print(traceback.format_exc())
        return None
    return vector_db
    # --- End FAISS Change ---

def get_conversational_qa_chain(vector_db):
    """Creates the Conversational QA Chain (works with FAISS or Chroma)."""
    llm = None
    if USE_GOOGLE_AI:
        st.info("Initializing Google AI LLM (gemini-1.5-flash-latest)...")
        try:
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=1, convert_system_message_to_human=True)
            st.info("Google AI LLM initialized.")
        except Exception as e: st.error(f"Error initializing LLM: {e}."); return None
    if llm is None: st.error("LLM not initialized."); return None

    # FAISS vectorstore object also has .as_retriever() method
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
        llm=llm, retriever=retriever, memory=memory, return_source_documents=True,
        output_key='answer', combine_docs_chain_kwargs={"prompt": QA_PROMPT}
    )
    return conversational_chain

# --- Streamlit UI ---
def main():
    load_dotenv()
    st.set_page_config(page_title="DocuMind Q&A (FAISS + Google AI)", layout="wide")
    st.title("📄 DocuMind Q&A")

    model_name_display = "Google AI"
    parser_name_display = "PyPDF"
    storage_type = "FAISS (In-Memory)" # Updated storage type
    st.caption(f"✨ Powered by {model_name_display} | Parser: {parser_name_display} | Storage: {storage_type} | Conversational Mode")

    # Initialize session state
    if "vector_db" not in st.session_state: st.session_state.vector_db = None
    if "qa_chain" not in st.session_state: st.session_state.qa_chain = None
    if "processed_files_info" not in st.session_state: st.session_state.processed_files_info = None
    if "messages" not in st.session_state: st.session_state.messages = []

    with st.sidebar:
        st.header("📁 Document Setup")
        st.info(f"Mode: **{model_name_display}** / **{parser_name_display}** / **{storage_type}**")
        st.warning("Vector store is In-Memory. PDFs must be re-processed if app restarts.")

        uploaded_files = st.file_uploader(
            "Upload PDF files", type="pdf", accept_multiple_files=True, key="pdf_uploader_faiss_v1"
        )

        # Simplified button logic - always processes uploaded files into memory
        if st.button(f"Process Uploaded PDF(s)", key="process_button_faiss_v1", disabled=not uploaded_files):
            with st.spinner(f"Processing PDFs into memory (FAISS)..."):

                # 1. Always start fresh for session state when button is clicked
                st.session_state.vector_db = None
                st.session_state.qa_chain = None
                st.session_state.messages = []
                st.session_state.processed_files_info = None

                # 2. Process PDFs and create a new IN-MEMORY FAISS vector store
                vector_db_obj = load_and_process_pdfs(uploaded_files)

                # 3. Update session state if processing was successful
                if vector_db_obj:
                    st.session_state.vector_db = vector_db_obj
                    st.session_state.qa_chain = get_conversational_qa_chain(st.session_state.vector_db)
                    st.session_state.processed_files_info = {
                        "names": [f.name for f in uploaded_files],
                        "model_type": model_name_display,
                        "parser": parser_name_display,
                        "storage": storage_type
                    }
                    st.success(f"PDFs processed (in memory)! Ready to chat.")
                else:
                    st.session_state.qa_chain = None
                    st.session_state.processed_files_info = None
                    st.error(f"PDF processing failed. In-memory vector store could not be created.")

        if st.session_state.processed_files_info:
            parser_info = f"(Parser: {st.session_state.processed_files_info.get('parser', 'Unknown')})"
            storage_info = f"(Storage: {st.session_state.processed_files_info.get('storage', 'FAISS')})" # Updated
            st.success(f"Active Docs ({st.session_state.processed_files_info['model_type']}{parser_info}{storage_info}):")
            for name in st.session_state.processed_files_info["names"]: st.markdown(f"- `{name}`")

        if st.button("Clear Chat History", key="clear_chat_faiss_v1"):
            st.session_state.messages = []
            st.rerun()

    st.header("💬 Chat with Your Documents")
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=True)
            if "sources" in message and message["sources"]:
                with st.expander("Sources Used"):
                    for src in message["sources"]: st.markdown(f"- `{src}`")

    # Handle chat input
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
