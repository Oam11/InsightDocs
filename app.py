import streamlit as st
import os
from utils import DocumentProcessor
import tempfile
import uuid
from datetime import datetime
import json
import hashlib

# Constants for local storage
CONFIG_DIR = os.path.expanduser("~/.insightdocs")
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")

def ensure_config_dir():
    """Ensure the config directory exists."""
    if not os.path.exists(CONFIG_DIR):
        os.makedirs(CONFIG_DIR, exist_ok=True)

def save_api_key(api_key: str):
    """Save API key to local config file."""
    try:
        ensure_config_dir()
        config = {
            "api_key": api_key,
            "saved_date": datetime.now().isoformat()
        }
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f)
        return True
    except Exception as e:
        print(f"Error saving API key: {e}")
        return False

def load_api_key():
    """Load API key from local config file."""
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
                return config.get("api_key", "")
        return ""
    except Exception as e:
        print(f"Error loading API key: {e}")
        return ""

def clear_stored_api_key():
    """Clear the stored API key."""
    try:
        if os.path.exists(CONFIG_FILE):
            os.remove(CONFIG_FILE)
        return True
    except Exception as e:
        print(f"Error clearing API key: {e}")
        return False

# Get API key from stored config or user input
stored_api_key = load_api_key()
api_key = None

# Default model selection (can be overridden in sidebar)
default_model = "llama-3.3-70b-versatile"

# Try to use stored API key first
if stored_api_key and stored_api_key.startswith("gsk_"):
    api_key = stored_api_key
    show_api_input = False
else:
    show_api_input = True

# If no valid stored key, ask user for input
if show_api_input:
    st.markdown("### 🔑 API Key Setup")
    st.info("Please enter your Groq API key. You can get one free at [console.groq.com](https://console.groq.com)")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        user_api_key = st.text_input(
            "Groq API Key:",
            type="password",
            placeholder="Enter your Groq API key (starts with 'gsk_')",
            help="Your API key will be stored locally for convenience.",
            key="api_key_input"
        )
    
    with col2:
        save_key = st.checkbox("💾 Remember", value=True, help="Save API key locally so you don't have to enter it again")
    
    if user_api_key:
        if user_api_key.startswith("gsk_"):
            api_key = user_api_key
            if save_key:
                if save_api_key(api_key):
                    st.success("✅ API key saved locally!")
                    st.rerun()
                else:
                    st.warning("⚠️ Could not save API key locally, but will use for this session")
        else:
            st.error("❌ Invalid Groq API key format. The API key should start with 'gsk_'")
            st.stop()
    else:
        st.warning("⚠️ Please enter your Groq API key to continue.")
        st.markdown("""
        **How to get your API key:**
        1. Go to [console.groq.com](https://console.groq.com)
        2. Sign up for a free account
        3. Generate an API key
        4. Copy and paste it above
        5. Click "Remember" to save it for future sessions
        6. The api key is stored locally on your machine
        """)
        st.stop()

# Initialize session state
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if 'current_api_key' not in st.session_state:
    st.session_state.current_api_key = None
if 'current_model' not in st.session_state:
    st.session_state.current_model = default_model
if 'qa_history' not in st.session_state:
    st.session_state.qa_history = []
if 'processed_docs' not in st.session_state:
    st.session_state.processed_docs = False
if 'document_store' not in st.session_state:
    st.session_state.document_store = None
if 'last_question' not in st.session_state:
    st.session_state.last_question = ""

# Set page config
st.set_page_config(
    page_title="InsightDocs - Multi-Format RAG",
    page_icon="📚",
    layout="wide"
)

# Title and description
st.title("📚 InsightDocs - Multi-Format RAG Application")
st.markdown("""
This application allows you to chat with your documents and images using advanced RAG (Retrieval-Augmented Generation) technology. 
Upload documents, images, or mixed file types and ask questions about their content.

**Supported formats:** PDF, Word, Excel, PowerPoint, Text files, Images (with OCR), and more!
""")

# Sidebar for file upload
with st.sidebar:
    # API Key status
    st.markdown("### 🔑 API Status")
    if api_key:
        masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else api_key[:4] + "..."
        if stored_api_key:
            st.success(f"✅ Stored Locally: `{masked_key}`")
        else:
            st.info(f"🔒 Session Only: `{masked_key}`")
        
        # API Key management options
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Change Key", help="Enter a new API key"):
                clear_stored_api_key()
                st.rerun()
        
        with col2:
            if stored_api_key and st.button("🗑️ Clear Stored", help="Remove saved API key"):
                if clear_stored_api_key():
                    st.success("✅ Stored API key cleared!")
                    st.rerun()
                else:
                    st.error("❌ Failed to clear stored key")
    else:
        st.error("❌ No API key provided")
    
    st.markdown("---")
    
    # Model Selection
    st.header("🤖 Model Configuration")
    model_options = {
        "Llama 3.3 70B (Recommended)": "llama-3.3-70b-versatile",
        "Llama 3.1 70B": "llama-3.1-70b-versatile", 
        "Llama 3.1 8B": "llama-3.1-8b-instant",
        "Llama 3.2 90B": "llama-3.2-90b-text-preview",
        "Llama 3.2 11B": "llama-3.2-11b-text-preview",
        "Llama 3.2 3B": "llama-3.2-3b-preview",
        "Llama 3.2 1B": "llama-3.2-1b-preview"
    }
    
    selected_model_name = st.selectbox(
        "Choose AI Model:",
        options=list(model_options.keys()),
        index=0,
        help="Select the language model for document analysis. Larger models generally provide better responses but may be slower."
    )
    
    selected_model = model_options[selected_model_name]
    
    st.markdown("---")
    st.header("📁 Upload Documents & Images")
    st.markdown("*Drag & drop or browse files*")
    uploaded_files = st.file_uploader(
        "Choose your documents",
        type=["pdf", "docx", "txt", "csv", "json", "xml", "yml", "yaml", "xls", "xlsx", "pptx", "html", "md", "jpg", "jpeg", "png", "bmp", "tiff", "tif", "webp", "gif"],
        accept_multiple_files=True,
        help="Supported formats: Documents (PDF, Word, Excel, PowerPoint, etc.), Images (JPG, PNG, TIFF, etc.), and Text files"
    )
    
    if uploaded_files:
        if st.button("Process Documents"):
            with st.spinner("Processing documents..."):
                try:
                    # Save uploaded files temporarily with proper extensions
                    temp_files = []
                    file_mapping = {}  # Map temp files to original names
                    
                    for uploaded_file in uploaded_files:
                        # Get file extension from original filename
                        file_extension = os.path.splitext(uploaded_file.name)[1]
                        
                        # Create temp file with proper extension
                        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
                        temp_file.write(uploaded_file.getvalue())
                        temp_file.close()
                        
                        temp_files.append(temp_file.name)
                        file_mapping[temp_file.name] = uploaded_file.name
                        
                        print(f"📁 Saved {uploaded_file.name} as {temp_file.name}")
                    
                    # Store the mapping for better error reporting
                    st.session_state.file_mapping = file_mapping
                    
                    # Process documents
                    st.session_state.document_store = st.session_state.doc_processor.process_documents(
                        temp_files, 
                        file_mapping
                    )
                    st.session_state.processed_docs = True
                    
                    # Display processed files with file type info
                    metadata = st.session_state.document_store['metadata']
                    
                    if metadata['processing_errors']:
                        st.warning(f"⚠️ Some files could not be processed ({len(metadata['processing_errors'])} errors)")
                        with st.expander("� View Processing Details"):
                            for error in metadata['processing_errors']:
                                st.text(f"❌ {error}")
                    
                    if metadata['total_files'] > 0:
                        st.success(f"✅ Successfully processed {metadata['total_files']} files!")
                        
                        st.markdown("### 📊 Processing Summary:")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Files", metadata['total_files'])
                        with col2:
                            st.metric("Total Chunks", metadata['total_chunks'])
                        with col3:
                            st.metric("File Types", len(metadata['file_types']))
                        
                        st.markdown("### 📄 Processed Files:")
                        for file_name, file_type in metadata['file_types'].items():
                            if 'Image' in file_type:
                                st.markdown(f"🖼️ **{file_name}** - {file_type} (OCR processed)")
                            else:
                                st.markdown(f"📄 **{file_name}** - {file_type}")
                    else:
                        st.error("❌ No files could be processed successfully")
                    
                    # Clean up temporary files
                    for temp_file in temp_files:
                        os.unlink(temp_file)
                    
                except ValueError as e:
                    st.error(f"Error processing documents: {str(e)}")
                    st.info("Please make sure your files contain readable text content.")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {str(e)}")
                    st.info("Please try uploading different files or check the file formats.")
                finally:
                    # Clean up temporary files in case of error
                    for temp_file in temp_files:
                        try:
                            os.unlink(temp_file)
                        except:
                            pass
    
    # Session information
    st.markdown("---")
    st.markdown(f"**Session ID:** {st.session_state.session_id}")
    
    # Download Q&A PDF
    if st.session_state.qa_history:
        if st.button("📄 Download Q&A PDF", help="Download your complete Q&A session as a PDF"):
            with st.spinner("📝 Generating PDF..."):
                try:
                    # Create a temporary file for the PDF
                    temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
                    temp_pdf.close()
                    
                    # Generate PDF with the session history
                    pdf_path = st.session_state.doc_processor.generate_qa_pdf(
                        temp_pdf.name, 
                        st.session_state.qa_history
                    )
                    
                    # Read the PDF file
                    with open(pdf_path, 'rb') as pdf_file:
                        pdf_data = pdf_file.read()
                    
                    # Create download button
                    st.download_button(
                        label="💾 Click to Download PDF",
                        data=pdf_data,
                        file_name=f"InsightDocs_QA_Session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        help="Your complete Q&A session in PDF format"
                    )
                    
                    st.success("✅ PDF generated successfully!")
                    
                    # Clean up temporary file
                    try:
                        os.unlink(pdf_path)
                    except:
                        pass
                        
                except Exception as e:
                    st.error(f"❌ Error generating PDF: {str(e)}")
    else:
        st.info("💡 Ask some questions first to generate a Q&A PDF report")

# Initialize or update DocumentProcessor based on selected model and API key
need_reinit = (
    'doc_processor' not in st.session_state or 
    st.session_state.current_api_key != api_key or
    st.session_state.current_model != selected_model
)

if need_reinit:
    st.session_state.doc_processor = DocumentProcessor(api_key, selected_model)
    st.session_state.current_api_key = api_key
    st.session_state.current_model = selected_model
    # Reset processing state when API key or model changes
    if ('current_api_key' in st.session_state and st.session_state.current_api_key != api_key) or \
       ('current_model' in st.session_state and st.session_state.current_model != selected_model):
        st.session_state.processed_docs = False
        st.session_state.document_store = None
        st.session_state.qa_history = []

# Main chat interface
if not st.session_state.processed_docs:
    st.info("👆 Please upload and process your documents first.")
    st.markdown("""
    ### 🎯 What you can do:
    - Upload **multiple file types** at once
    - Ask questions about **text documents** (PDF, Word, Excel, etc.)
    - Extract text from **images** using OCR technology
    - Get **source references** for every answer
    - Download your **Q&A session** as a PDF
    """)
else:
    # Create a container for the chat history
    chat_container = st.container()
    
    # Display chat history in the container
    with chat_container:
        for qa in st.session_state.qa_history:
            st.markdown(f"**Q:** {qa['question']}")
            st.markdown(f"**A:** {qa['answer']}")
            st.markdown(f"*Asked at: {qa['timestamp']}*")
            st.markdown("---")
    
    # Create a form for the question input
    with st.form("question_form"):
        user_question = st.text_input("Ask a question about your documents:")
        submit_button = st.form_submit_button("Ask Question")
        
        if submit_button and user_question:
            with st.spinner("🔍 Searching through your documents..."):
                # Get answer from the enhanced QA chain
                qa_chain = st.session_state.doc_processor.create_qa_chain(st.session_state.document_store)
                result = qa_chain.invoke({"query": user_question})
                answer = result['result']
                
                # Display source information if available
                if 'source_documents' in result and result['source_documents']:
                    with st.expander("📚 Source Documents Used"):
                        for i, doc in enumerate(result['source_documents'][:3]):  # Show top 3 sources
                            source = doc.metadata.get('source', 'Unknown')
                            content_type = doc.metadata.get('content_type', 'Unknown')
                            chunk_id = doc.metadata.get('chunk_id', 'N/A')
                            st.markdown(f"**Source {i+1}:** {source} ({content_type}) - Chunk {chunk_id}")
                            st.markdown(f"*Preview:* {doc.page_content[:200]}...")
                
                # Add to history
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.qa_history.append({
                    'question': user_question,
                    'answer': answer,
                    'timestamp': timestamp
                })
                st.session_state.doc_processor.add_qa_to_history(user_question, answer)
                
                # Store the last question
                st.session_state.last_question = user_question
                
                # Update the display
                st.rerun()

# Footer
st.markdown("---")
st.markdown("🚀 **InsightDocs** - Enhanced RAG with OCR & Multi-format Support") 
