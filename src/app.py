"""
Streamlit application for chatting with documents using RAG (Retrieval Augmented Generation).

This application allows users to select a folder containing documents, processes them into
a vector store, and enables question-answering based on the document content using an LLM.
"""

import os
import time

import streamlit as st

from document_loader import load_txt, load_pdf, load_docx, load_odt
from scan_folders import scan_folders
from vector_store import (
    chunk_text,
    create_collection,
    add_chunks,
    create_file_index_chunk,
)

from response_generator import set_llm, generate_answer, set_langchain_history
from retrieval_system import query_documents

from langchain_core.messages import HumanMessage, AIMessage

# Map file extensions to their respective loader functions
loaders = {".txt": load_txt, ".pdf": load_pdf, ".docx": load_docx, ".odt": load_odt}


# ============================================================================
# Helper Functions
# ============================================================================


def is_streamlit_cloud():
    """Check if running on Streamlit Cloud"""
    #return os.getenv("STREAMLIT_SHARING_MODE") is not None
    env = os.environ
    return (
            "STREAMLIT_RUNTIME" in env or
            "STREAMLIT_SHARING_MODE" in env or
            "STREAMLIT_SERVER_HEADLESS" in env
    )


def get_secret(key: str, default: str = None):
    """
    Unified way to fetch secrets.
    Uses st.secrets on Streamlit Cloud, .env locally.
    """
    if is_streamlit_cloud():
        return st.secrets.get(key, default)
    else:
        load_dotenv()
        return os.getenv(key, default)


def extract_zip_and_scan(uploaded_zip):
    """
    Extract uploaded ZIP file to temporary directory and scan for documents.

    Args:
        uploaded_zip: Streamlit UploadedFile object containing ZIP data.

    Returns:
        tuple: (temp_dir_path, list of file paths)
    """
    import tempfile
    import zipfile

    # Create temporary directory
    temp_dir = tempfile.mkdtemp()

    # Extract ZIP to temp directory
    with zipfile.ZipFile(uploaded_zip, "r") as zip_ref:
        zip_ref.extractall(temp_dir)

    # Scan the extracted folder for documents
    files = scan_folders(temp_dir)

    return temp_dir, files


def render_sidebar():
    """
    Render the sidebar with document settings and controls.

    Handles two modes:
    - Cloud mode: ZIP file upload for document submission
    - Local mode: Folder path input for direct filesystem access

    Also provides chat history clearing functionality.
    """

    with st.sidebar:
        st.title("📁 Document Settings")

        # Cloud mode: Use ZIP upload since filesystem access is restricted
        if is_streamlit_cloud():
            uploaded_zip = st.file_uploader(
                "Upload a ZIP file with your documents:",
                type=["zip"],
                help="Upload a ZIP file containing your documents (PDF, TXT, DOCX, ODT)",
            )

            if uploaded_zip is not None:
                if st.button("Load ZIP", use_container_width=True):
                    st.session_state.uploaded_zip = uploaded_zip
                    # Clear collection to trigger re-indexing
                    if "collection" in st.session_state:
                        del st.session_state.collection
                    if "files" in st.session_state:
                        del st.session_state.files
                    st.rerun()

            # Show upload status
            if "uploaded_zip" in st.session_state:
                st.markdown("---")
                st.success("📦 **ZIP Uploaded:**")
                st.code(st.session_state.uploaded_zip.name, language=None)

                if "files" in st.session_state:
                    st.info(f"📄 **Files Indexed:** {len(st.session_state.files)}")
        else:
            # Local mode: Direct folder path access
            # Input field for folder path
            folder_path = st.text_input(
                "Enter folder path:",
                placeholder="C:/Users/YourName/Documents or /home/user/docs",
                help="Enter the full path to your documents folder",
            )

            if st.button("Load Folder", use_container_width=True):
                if folder_path and os.path.exists(folder_path):
                    st.session_state.folder_path = folder_path
                    # Clear collection to trigger re-indexing
                    if "collection" in st.session_state:
                        del st.session_state.collection
                    if "files" in st.session_state:
                        del st.session_state.files
                    st.rerun()
                elif folder_path:
                    st.error("Folder not found. Please check the path.")
                    if "files" in st.session_state:
                        del st.session_state.files
                    if "folder_path" in st.session_state:
                        del st.session_state.folder_path
                else:
                    st.info("Please enter a folder path")

            # Show current folder if selected
            if "folder_path" in st.session_state:
                st.markdown("---")
                st.success("📂 **Current Folder:**")
                st.code(st.session_state.folder_path, language=None, wrap_lines=True)

                # Show number of files if available
                if "files" in st.session_state:
                    st.info(f"📄 **Files Indexed:** {len(st.session_state.files)}")

        # Clear chat button
        st.markdown("---")
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


def initialize_vector_store(folder_path, files):
    """
    Initialize and populate the vector store with document chunks.

    Args:
        folder_path (str): Path to the folder containing documents.
        files (list): List of file paths to process and index.
    """
    if "collection" not in st.session_state:
        try:
            collection = create_collection(folder_path)
        except Exception:
            st.write("Error setting up the document storage system.")
            st.stop()

        with st.spinner("Indexing documents..."):
            # Create progress bar
            progress_bar = st.progress(0)

            # Process each document file
            for i, file in enumerate(files):
                # Show progress
                progress_bar.progress((i + 1) / len(files))
                _, extension = os.path.splitext(file)

                # Load document content using the appropriate loader
                try:
                    fn = loaders[extension]
                    content = fn(file)
                except KeyError:
                    content = ""
                    st.write(f"File {file} not supported. Skipping.")

                # Split content into chunks and add to vector store
                chunks = chunk_text(content, file)
                if chunks:
                    try:
                        collection = add_chunks(chunks, collection)
                    except Exception:
                        st.write(f"Error processing file {file}. Skipping.")

            # Create and add a special index of all file names for better retrieval
            file_index = create_file_index_chunk(files)
            try:
                collection = add_chunks(file_index, collection)
            except Exception:
                st.write(
                    "Warning: Could not index file names. You can still search document content."
                )

            st.session_state.collection = collection
            # Reload page after indexing
            st.rerun()
    else:
        collection = st.session_state.collection


def handle_chat_input(collection, llm):
    """
    Handle user chat input, retrieve relevant documents, and generate responses.

    Args:
        collection: ChromaDB collection containing indexed documents.
        llm: Language model instance for generating answers.
    """
    user_input = st.chat_input("Ask your question")
    if user_input:

        # Add and display user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Retrieve relevant document chunks based on the user's question
        try:
            related_chunks = query_documents(
                collection=collection, query_text=user_input
            )
        except Exception:
            st.info("Error searching documents. Please try again.")
            st.stop()

        # Generate LLM response using retrieved context and conversation history
        history = set_langchain_history(st.session_state.messages)
        answer = generate_answer(llm, user_input, related_chunks, history)

        # Add and display assistant response
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)


# ============================================================================
# UI Setup and Folder Selection
# ============================================================================

st.title("Chat with your documents!")
st.write(time.strftime("%d %b, %Y"))

# Sidebar
render_sidebar()

if "folder_path" in st.session_state:
    # Local mode: Scan the folder for supported document files
    files = scan_folders(st.session_state.folder_path)
    if not files:
        if "show_popup" not in st.session_state:
            st.session_state.show_popup = True
        if st.session_state.show_popup:
            st.info("No documents found")
        st.stop()

    st.session_state.files = files
elif "uploaded_zip" in st.session_state:
    # Cloud mode: extract ZIP and scan
    temp_dir, files = extract_zip_and_scan(st.session_state.uploaded_zip)
    if not files:
        st.info("No documents found in ZIP")
        st.stop()
    st.session_state.files = files
    st.session_state.temp_dir = temp_dir
else:
    st.stop()

# ============================================================================
# Document Processing and Vector Store Creation
# ============================================================================

# Create a vector store collection specific to the selected folder
# Use ZIP name for cloud, folder path for local
if is_streamlit_cloud():
    collection_path = st.session_state.uploaded_zip.name
else:
    collection_path = st.session_state.folder_path

initialize_vector_store(collection_path, st.session_state.files)

# ============================================================================
# Chat Interface and Question Answering
# ============================================================================

# Initialize LLM and conversation history
if "llm" not in st.session_state:
    st.session_state.llm = set_llm()
llm = st.session_state.llm

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle new user input
handle_chat_input(st.session_state.collection, llm)
