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
import streamlit as st
import os
from dotenv import load_dotenv
import shutil

# LangChain components
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate # Added for custom prompt

# --- CHOOSE YOUR MODELS ---
USE_GOOGLE_AI = True
USE_OPENAI = False
USE_LOCAL_MODELS = False

if USE_GOOGLE_AI:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from langchain_google_genai import ChatGoogleGenerativeAI
elif USE_OPENAI:
    from langchain_openai import OpenAIEmbeddings
    from langchain_openai import ChatOpenAI
# elif USE_LOCAL_MODELS: # Placeholder
    # pass

# --- Configuration ---
VECTOR_STORE_DIR = "google_ai_vector_store"

# --- Core Functions ---

def clear_vector_store():
    if os.path.exists(VECTOR_STORE_DIR):
        try:
            shutil.rmtree(VECTOR_STORE_DIR)
            st.sidebar.info(f"Cleared old vector store: '{VECTOR_STORE_DIR}'")
        except Exception as e:
            st.sidebar.error(f"Error clearing vector store: {e}")

def load_and_process_pdfs(pdf_files, chunk_size=1500, chunk_overlap=300): # Slightly increased chunk size
    documents = []
    temp_upload_dir = "temp_pdf_uploads"
    os.makedirs(temp_upload_dir, exist_ok=True)
    for pdf_file in pdf_files:
        temp_file_path = os.path.join(temp_upload_dir, pdf_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(pdf_file.getbuffer())
        loader = PyPDFLoader(temp_file_path)
        try:
            documents.extend(loader.load())
        except Exception as e:
            st.error(f"Error loading PDF '{pdf_file.name}': {e}. Skipping.")
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
    if not documents:
        st.warning("No text extracted from PDF(s).")
        return None
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = text_splitter.split_documents(documents)
    embeddings_model = None
    if USE_GOOGLE_AI:
        st.info("Using Google AI Embeddings (models/embedding-001)...")
        try:
            embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            st.info("Google AI Embeddings initialized.")
        except Exception as e:
            st.error(f"Error initializing Google AI Embeddings: {e}.")
            return None
    # Add other model logic here if needed
    if embeddings_model is None: return None
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    st.info("Creating vector embeddings (Google AI)...")
    try:
        vector_db = Chroma.from_documents(
            documents=split_docs, embedding=embeddings_model, persist_directory=VECTOR_STORE_DIR
        )
        st.info("Vector store created/updated.")
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None
    return vector_db

def get_conversational_qa_chain(vector_db):
    llm = None
    if USE_GOOGLE_AI:
        st.info("Initializing Google AI LLM (gemini-1.5-flash-latest)...")
        try:
            # Using gemini-1.5-flash-latest, temperature lowered for more factual responses
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=1, convert_system_message_to_human=True)
            st.info("Google AI LLM initialized.")
        except Exception as e:
            st.error(f"Error initializing Google AI LLM: {e}.")
            return None
    # Add other model logic here
    if llm is None: return None

    retriever = vector_db.as_retriever(search_kwargs={"k": 4}) 
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True, output_key='answer'
    )

    # --- CUSTOM PROMPT for combining documents ---
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
    # --- END OF CUSTOM PROMPT ---

    conversational_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        output_key='answer',
        combine_docs_chain_kwargs={"prompt": QA_PROMPT} # Using the custom prompt
    )
    return conversational_chain

# --- Streamlit UI ---
def main():
    load_dotenv()
    st.set_page_config(page_title="DocuMind Q&A (Google AI)", layout="wide")
    st.title("📄 DocuMind Q&A")

    model_name_display = "Google AI" if USE_GOOGLE_AI else ("OpenAI" if USE_OPENAI else "Local Models")
    st.caption(f"✨ Powered by {model_name_display} | Conversational Mode")

    if "vector_db" not in st.session_state: st.session_state.vector_db = None
    if "qa_chain" not in st.session_state: st.session_state.qa_chain = None
    if "processed_files_info" not in st.session_state: st.session_state.processed_files_info = None
    if "messages" not in st.session_state: st.session_state.messages = []

    with st.sidebar:
        st.header("📁 Document Setup")
        st.info(f"Current Mode: *{model_name_display}*")
        uploaded_files = st.file_uploader(
            "Upload PDF files", type="pdf", accept_multiple_files=True, key="pdf_uploader_google"
        )
        force_reprocess = st.checkbox("Force re-process & clear old vector store", key="reprocess_google")

        if st.button("Process Uploaded PDF(s)", key="process_button_google", disabled=not uploaded_files):
            with st.spinner(f"Processing PDFs with {model_name_display}..."):
                if force_reprocess:
                    clear_vector_store()
                    st.session_state.vector_db = None
                    st.session_state.qa_chain = None
                    st.session_state.messages = []

                load_existing = False
                if not force_reprocess and os.path.exists(VECTOR_STORE_DIR) and st.session_state.processed_files_info:
                    if st.session_state.processed_files_info.get("model_type") == model_name_display:
                        load_existing = True
                
                if load_existing:
                    st.info("Attempting to load existing vector store...")
                    try:
                        # This assumes the same embedding model is used if model_name_display matches
                        if USE_GOOGLE_AI:
                            embeddings_model_loader = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                        # Add other model loaders here
                        st.session_state.vector_db = Chroma(persist_directory=VECTOR_STORE_DIR, embedding_function=embeddings_model_loader)
                        st.info("Loaded existing vector store.")
                    except Exception as e:
                        st.warning(f"Could not load existing vector store: {e}. Re-processing.")
                        clear_vector_store()
                        st.session_state.vector_db = None
                        load_existing = False

                if not load_existing or st.session_state.vector_db is None:
                    if st.session_state.vector_db is None and not force_reprocess and os.path.exists(VECTOR_STORE_DIR):
                        clear_vector_store()
                    st.session_state.vector_db = load_and_process_pdfs(uploaded_files)
                
                if st.session_state.vector_db:
                    st.session_state.qa_chain = get_conversational_qa_chain(st.session_state.vector_db)
                    st.session_state.processed_files_info = {
                        "names": [f.name for f in uploaded_files], "model_type": model_name_display
                    }
                    if force_reprocess or not load_existing:
                         st.session_state.messages = []
                    st.success("PDFs processed! Ready to chat.")
                else:
                    st.error("PDF processing failed.")
        
        if st.session_state.processed_files_info:
            st.success(f"Active Docs ({st.session_state.processed_files_info['model_type']}):")
            for name in st.session_state.processed_files_info["names"]: st.markdown(f"- {name}")
        
        if st.button("Clear Chat History", key="clear_chat_final"):
            st.session_state.messages = []
            st.rerun()

    st.header("💬 Chat with Your Documents")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("Sources Used"):
                    for src in message["sources"]: st.markdown(f"- {src}")

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
                        msg_placeholder.markdown(answer)
                        if sources:
                            with st.expander("Sources Used"):
                                for src in sources: st.markdown(f"- {src}")
                        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
                    except Exception as e:
                        err_msg = f"An error occurred with Google AI: {e}"
                        st.error(err_msg)
                        st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}", "sources": []})
                        import traceback
                        print(f"ERROR in QA Chain: {traceback.format_exc()}")
                        msg_placeholder.markdown("Sorry, an error occurred. Check terminal logs.")
    else:
        st.info("Please upload and process PDF files to start chatting.")

if __name__ == "__main__":
    main()
