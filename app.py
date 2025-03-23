import streamlit as st
import os
from utils import DocumentProcessor
import tempfile
import uuid
from datetime import datetime

# Get API key from Streamlit secrets
try:
    api_key = st.secrets["GROQ_API_KEY"]
    if not api_key or api_key == "your_api_key_here":
        st.error("""
        Please set your actual Groq API key in `.streamlit/secrets.toml`. 
        The current value is either empty or still using the placeholder.
        
        1. Go to https://console.groq.com to get your API key
        2. Open `.streamlit/secrets.toml`
        3. Replace the placeholder with your actual API key:
        ```toml
        GROQ_API_KEY = "your_actual_groq_api_key_here"
        ```
        """)
        st.stop()
except FileNotFoundError:
    st.error("""
    No secrets.toml file found. Please create a `.streamlit/secrets.toml` file in your project directory with your Groq API key:
    
    ```toml
    GROQ_API_KEY = "your_actual_groq_api_key_here"
    ```
    
    You can get your API key from https://console.groq.com
    """)
    st.stop()
except KeyError:
    st.error("""
    GROQ_API_KEY not found in secrets.toml. Please add your Groq API key to the `.streamlit/secrets.toml` file:
    
    ```toml
    GROQ_API_KEY = "your_actual_groq_api_key_here"
    ```
    
    You can get your API key from https://console.groq.com
    """)
    st.stop()

# Add basic API key validation
if not api_key.startswith("gsk_"):
    st.error("""
    Invalid Groq API key format. The API key should start with 'gsk_'.
    Please check your API key in `.streamlit/secrets.toml` and make sure you're using the correct key from https://console.groq.com
    """)
    st.stop()

# Initialize session state
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if 'doc_processor' not in st.session_state:
    st.session_state.doc_processor = DocumentProcessor(api_key)
if 'qa_history' not in st.session_state:
    st.session_state.qa_history = []
if 'processed_docs' not in st.session_state:
    st.session_state.processed_docs = False
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'last_question' not in st.session_state:
    st.session_state.last_question = ""

# Set page config
st.set_page_config(
    page_title="Document Chat with Groq",
    page_icon="📚",
    layout="wide"
)

# Title and description
st.title("Document Chat with Groq")
st.markdown("""
This application allows you to chat with your documents using Groq's Gemma2 model. 
Upload your documents and ask questions about their content.
""")

# Sidebar for file upload
with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose your documents",
        type=["pdf", "docx", "txt", "csv", "json", "xml", "yml", "yaml", "xls", "xlsx", "pptx", "html"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if st.button("Process Documents"):
            with st.spinner("Processing documents..."):
                try:
                    # Save uploaded files temporarily
                    temp_files = []
                    for uploaded_file in uploaded_files:
                        temp_file = tempfile.NamedTemporaryFile(delete=False)
                        temp_file.write(uploaded_file.getvalue())
                        temp_files.append(temp_file.name)
                    
                    # Process documents
                    st.session_state.vector_store = st.session_state.doc_processor.process_documents(temp_files)
                    st.session_state.processed_docs = True
                    
                    # Display processed files
                    st.success("Documents processed successfully!")
                    st.markdown("### Processed Documents:")
                    for file in uploaded_files:
                        st.markdown(f"- {file.name}")
                    
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
        if st.button("Download Q&A PDF"):
            with st.spinner("Generating PDF..."):
                try:
                    # Create a temporary file for the PDF
                    temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
                    pdf_path = st.session_state.doc_processor.generate_qa_pdf(temp_pdf.name)
                    
                    # Read the PDF file
                    with open(pdf_path, 'rb') as pdf_file:
                        pdf_data = pdf_file.read()
                    
                    # Create download button
                    st.download_button(
                        label="Click to Download PDF",
                        data=pdf_data,
                        file_name=f"qa_session_{st.session_state.session_id}.pdf",
                        mime="application/pdf"
                    )
                    
                    # Clean up temporary file after a short delay
                    st.session_state['pdf_to_delete'] = pdf_path
                except Exception as e:
                    st.error(f"Error generating PDF: {str(e)}")
                finally:
                    # Clean up the temporary file in the next run
                    if 'pdf_to_delete' in st.session_state:
                        try:
                            os.unlink(st.session_state['pdf_to_delete'])
                            del st.session_state['pdf_to_delete']
                        except:
                            pass

# Main chat interface
if not st.session_state.processed_docs:
    st.warning("Please upload and process your documents first.")
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
            with st.spinner("Generating answer..."):
                # Get answer from the model
                qa_chain = st.session_state.doc_processor.create_qa_chain(st.session_state.vector_store)
                answer = qa_chain.invoke({"query": user_question})['result']
                
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
st.markdown("Powered by Groq's Gemma2 model") 