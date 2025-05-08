# DOCUMind-RAG-# Conversational PDF Q&A Chatbot with Google Gemini

This Streamlit application allows users to upload one or more PDF documents and engage in a conversation to ask questions based on the content within those documents. The answers are generated using Google's Gemini AI models, retrieving relevant information from the uploaded text.

**Live Demo:** [Link to Your Deployed Streamlit App] *(<- Replace this with the actual URL from Streamlit Community Cloud)*

*(Optional: Add a Screenshot)*
<!-- ![App Screenshot](link_to_your_screenshot.png) -->

## Features

*   **Multiple PDF Upload:** Upload one or more PDF documents simultaneously.
*   **Content-Based Q&A:** Ask questions and receive answers generated *only* from the information present in the uploaded documents.
*   **Conversational Interface:** Maintains chat history, allowing for follow-up questions within the context of the ongoing conversation.
*   **Source Attribution:** Displays the source PDF file(s) used to generate each answer.
*   **Powered by Google AI:** Utilizes Google Generative AI for both text embeddings (`models/embedding-001`) and language generation (`gemini-1.5-flash-latest`).
*   **Simple PDF Parsing:** Uses `PyPDFLoader` for straightforward text extraction from PDFs.
*   **Vector Store:** Employs ChromaDB for efficient similarity search to find relevant text chunks.

## Tech Stack

*   **Language:** Python 3.9+
*   **Web Framework/UI:** Streamlit
*   **LLM & Embeddings:** Google Generative AI (Gemini via `langchain-google-genai`)
*   **Core Framework:** LangChain
*   **PDF Parsing:** PyPDFLoader (`langchain-community`)
*   **Vector Database:** ChromaDB (`chromadb`)
*   **Text Splitting:** `RecursiveCharacterTextSplitter` (`langchain`)
*   **Chat Memory:** `ConversationBufferMemory` (`langchain`)
*   **Environment Variables:** `python-dotenv` (for local development)

## Setup and Installation (Local Development)

Follow these steps to run the application on your local machine.

**1. Prerequisites:**

*   Git ([Download Git](https://git-scm.com/downloads))
*   Python 3.9 or higher ([Download Python](https://www.python.org/downloads/))
*   Access to Google AI Studio and an API Key ([Google AI Studio](https://aistudio.google.com/))

**2. Clone the Repository:**

```bash
git clone [Your GitHub Repository URL]
cd [Your Repository Folder Name]
