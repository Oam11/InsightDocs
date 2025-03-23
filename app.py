import streamlit as st
import os
from dotenv import load_dotenv
from utils import DocumentProcessor
import tempfile

# Load environment variables
load_dotenv()

# Get API key from environment variable
api_key = os.getenv('GEMINI_API_KEY')

# Set page config
st.set_page_config(
    page_title="Document Chat with Gemini",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []

# Title and description
st.title("📚 Document Chat with Gemini")
st.markdown("""
This application allows you to upload multiple documents and chat with them using Google's Gemini AI.
Supported file types: PDF, DOCX, TXT, and CSV.
""")

# Sidebar for file upload
with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose files",
        type=['pdf', 'docx', 'txt', 'csv'],
        accept_multiple_files=True
    )

# Main content area
if api_key:
    if uploaded_files:
        # Process uploaded files
        if st.button("Process Documents"):
            with st.spinner("Processing documents..."):
                # Create temporary directory for uploaded files
                with tempfile.TemporaryDirectory() as temp_dir:
                    file_paths = []
                    for uploaded_file in uploaded_files:
                        file_path = os.path.join(temp_dir, uploaded_file.name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getvalue())
                        file_paths.append(file_path)
                    
                    # Initialize document processor
                    processor = DocumentProcessor(api_key)
                    
                    # Process documents and create QA chain
                    vector_store = processor.process_documents(file_paths)
                    st.session_state.qa_chain = processor.create_qa_chain(vector_store)
                    st.session_state.processed_files = [f.name for f in uploaded_files]
                    
                    st.success(f"Successfully processed {len(uploaded_files)} documents!")
    
    # Display processed files
    if st.session_state.processed_files:
        st.subheader("Processed Documents")
        for file_name in st.session_state.processed_files:
            st.write(f"- {file_name}")
    
    # Chat interface
    if st.session_state.qa_chain:
        st.subheader("Chat with Your Documents")
        user_question = st.text_input("Ask a question about your documents:")
        
        if user_question:
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.qa_chain.run(user_question)
                    st.write("Answer:", response)
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
else:
    st.error("""
    Gemini API key not found. Please create a `.env` file in the project root with your API key:
    
    ```
    GEMINI_API_KEY=your_api_key_here
    ```
    """)

# Footer
st.markdown("---")
st.markdown("Made with Streamlit and Google Gemini AI") 