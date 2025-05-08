# Conversational PDF Q&A Chatbot using Google Gemini & FAISS

This Streamlit application allows users to upload one or more PDF documents and engage in a conversation to ask questions based on the content within those documents. The answers are generated using Google's Gemini AI models, retrieving relevant information from the uploaded text using an in-memory FAISS vector store.

**Live Demo:** [https://himanshu-docu-mind-ojm9g2ucp7fxhn3zndvols.streamlit.app/] *(<- Replace this with the actual URL from Streamlit Community Cloud)*

![Main Page](https://github.com/user-attachments/assets/7a5eb48f-d555-4d8a-859d-cd41fb6114a9)

## Features

*   **Multiple PDF Upload:** Upload one or more PDF documents simultaneously.
*   **Content-Based Q&A:** Ask questions and receive answers generated *only* from the information present in the uploaded documents.
*   **Conversational Interface:** Maintains chat history using LangChain's memory modules, allowing for follow-up questions.
*   **Source Attribution:** Displays the source PDF file(s) used to generate each answer.
*   **Powered by Google AI:** Utilizes Google Generative AI for text embeddings (`models/embedding-001`) and language generation (`gemini-1.5-flash-latest`).
*   **Simple PDF Parsing:** Uses `PyPDFLoader` for basic text extraction.
*   **In-Memory Vector Store:** Employs FAISS for efficient in-memory similarity search. **Note:** Document embeddings are lost when the app restarts or the session ends due to the in-memory nature.

## Tech Stack

*   **Language:** Python 3.9+
*   **Web Framework/UI:** Streamlit
*   **LLM & Embeddings:** Google Generative AI (Gemini via `langchain-google-genai`)
*   **Core Framework:** LangChain (`langchain`, `langchain-community`)
*   **PDF Parsing:** PyPDFLoader (`langchain-community`, uses `pypdf`)
*   **Vector Database:** FAISS (`faiss-cpu`, `langchain-community`) - **In-Memory Only**
*   **Text Splitting:** `RecursiveCharacterTextSplitter` (`langchain`)
*   **Chat Memory:** `ConversationBufferMemory` (`langchain`)
*   **Environment Variables:** `python-dotenv` (for local development)
*   **Key Dependencies for Compatibility:**
    *   `protobuf==3.20.3` (Pinned to avoid Google library issues)
    *   `numpy<2.0.0` (Pinned to avoid `np.float_` errors in dependencies)
    *   `setuptools` (Explicitly included for Python 3.12 compatibility)

## Core Components Explained

This application integrates several key components to achieve the PDF Q&A functionality:

1.  **PDF Parsing & Chunking:**
    *   **Goal:** Extract usable text from uploaded PDFs and divide it into smaller, meaningful segments (chunks) suitable for processing by embedding models and LLMs.
    *   **Parsing (`PyPDFLoader`):** The application uses LangChain's `PyPDFLoader`. This loader reads the PDF file page by page and extracts the textual content. It's straightforward but primarily designed for text-based PDFs and may struggle with complex layouts, tables, or heavily image-based documents (including mathematical formulas, which may not render correctly). Each page is initially treated as a separate `Document`.
    *   **Chunking (`RecursiveCharacterTextSplitter`):** After loading, the text from the pages is further divided. This splitter attempts to break the text recursively using a list of separators (starting with double newlines, then single newlines, spaces, etc.) until the chunks are smaller than a defined `chunk_size` (e.g., 1500 characters). A `chunk_overlap` (e.g., 300 characters) is maintained between consecutive chunks to preserve contextual continuity. This character-based method is general-purpose but doesn't understand the semantic structure (paragraphs, sections) inherently.

2.  **Embedding & Storage in Vector DB:**
    *   **Goal:** Convert the text chunks into numerical representations (embeddings) that capture their semantic meaning, and store these embeddings efficiently for similarity searches.
    *   **Embedding (`GoogleGenerativeAIEmbeddings`):** Each text chunk is passed to the Google AI embedding model (`models/embedding-001`) via the `langchain-google-genai` integration. This API call converts the text into a high-dimensional vector. Similar concepts or text passages will have vectors that are "closer" together in the vector space. This requires a valid `GOOGLE_API_KEY`.
    *   **Storage (`FAISS` - In-Memory):** The generated text chunks and their corresponding embeddings are stored using LangChain's `FAISS` vector store integration. FAISS (Facebook AI Similarity Search) is a library optimized for efficient searching of dense vectors. In this application, FAISS is configured to run **entirely in memory**. This means:
        *   **Pros:** It's very fast and avoids all the filesystem permission/corruption issues encountered with disk-based persistence (like the ChromaDB errors) on ephemeral platforms like Streamlit Cloud.
        *   **Cons:** The entire index and all embeddings exist only in the RAM of the running application instance. When the app restarts, sleeps, or the user session ends, **this data is completely lost.** PDFs must be re-uploaded and re-processed for each new session/instance. The `faiss-cpu` package is used, meaning searches run on the CPU.

3.  **LLM-Powered Query Interface:**
    *   **Goal:** Handle user queries, retrieve relevant information from the vector store, maintain conversation context, and generate accurate, grounded answers using an LLM.
    *   **Retrieval (`FAISS.as_retriever()`):** When a user asks a question, their query is also converted into an embedding using the same Google AI embedding model. The FAISS vector store is then queried (via the `as_retriever()` method) to find the text chunks whose embeddings are most similar (e.g., top 4 chunks based on `k=4`) to the question embedding.
    *   **Memory (`ConversationBufferMemory`):** To enable follow-up questions, the history of the user's questions and the AI's answers is stored in memory for the current session.
    *   **Chain (`ConversationalRetrievalChain`):** This LangChain component orchestrates the process. It:
        1.  Takes the current question and the chat history from memory.
        2.  Potentially condenses the question and history into a standalone query suitable for retrieval.
        3.  Uses the retriever (FAISS) to fetch relevant document chunks based on the query.
        4.  Constructs a detailed prompt (using our custom `QA_PROMPT`) containing the chat history, the retrieved context chunks, and the current question.
        5.  Sends this prompt to the LLM.
    *   **LLM (`ChatGoogleGenerativeAI`):** The application uses Google's `gemini-1.5-flash-latest` model (or similar). This model receives the structured prompt from the chain and generates the final answer, guided by the instructions in the custom prompt to be detailed, accurate, and based only on the provided context. The `convert_system_message_to_human=True` parameter helps format the input for Gemini compatibility within this chain.
    *   **Output:** The generated answer and the sources (filenames of the retrieved chunks) are displayed to the user in the Streamlit chat interface.

