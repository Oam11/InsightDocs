# Document Chat with Groq

A Streamlit application that allows you to chat with your documents using Groq's Gemma2 model. This application implements a Retrieval Augmented Generation (RAG) model to provide accurate and context-aware responses based on your document content.

## Features

- **Multi-Document Support**: Upload and process multiple documents simultaneously
- **Session Management**: Each chat session is unique with a session ID
- **PDF Export**: Download your entire Q&A session as a PDF
- **Wide File Format Support**: Process various document types including:
  - PDF documents
  - Word documents (DOCX)
  - Text files (TXT)
  - Spreadsheets (CSV, XLS, XLSX)
  - JSON data
  - XML documents
  - YAML files
  - PowerPoint presentations (PPTX)
  - HTML documents
- **Smart Document Processing**:
  - Automatic content type detection
  - Document-specific context preservation
  - Improved chunking for better context understanding
- **Enhanced Q&A Experience**:
  - Powered by Groq's Gemma2 model
  - Context-aware responses
  - Document source tracking
  - Timestamp for each Q&A pair

## Prerequisites

- Python 3.8 or higher
- Groq API key

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Chat-with-documents.git
cd Chat-with-documents
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.streamlit/secrets.toml` file in your project directory:
```toml
GROQ_API_KEY = "your_api_key_here"
```

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Upload your documents using the file uploader in the sidebar
3. Click "Process Documents" to analyze the content
4. Start asking questions about your documents
5. Download the Q&A session as PDF when needed

## How It Works

1. **Document Processing**:
   - Documents are read and processed based on their file type
   - Content is split into chunks for efficient processing
   - Each chunk maintains context about its source document
   - Documents are indexed using FAISS for fast retrieval

2. **Question Answering**:
   - Questions are processed using Groq's Gemma2 model
   - Relevant document chunks are retrieved using semantic search
   - The model generates answers based on the retrieved context
   - Each Q&A pair is stored with timestamps

3. **Session Management**:
   - Each chat session has a unique ID
   - Q&A history is maintained throughout the session
   - Session data can be exported as PDF

4. **PDF Export**:
   - Generate a PDF containing all Q&A pairs
   - Includes timestamps and document sources
   - Nicely formatted with clear question-answer separation

## Technical Details

- **Model**: Groq's Gemma2 model for natural language understanding
- **Vector Store**: FAISS for efficient similarity search
- **Embeddings**: Sentence Transformers for document chunk representation
- **PDF Generation**: ReportLab for PDF creation and formatting

## Performance Considerations

- The Gemma2 model provides fast inference times
- Document chunking optimizes context window usage
- FAISS enables efficient similarity search
- Session state management ensures smooth user experience

## Troubleshooting

If you encounter any issues:

1. **API Key Issues**:
   - Ensure your API key is correctly set in the `.streamlit/secrets.toml` file
   - Verify the API key has the necessary permissions
   - Make sure the `.streamlit` directory exists in your project root

2. **File Processing Issues**:
   - Check if the file type is supported
   - Ensure the file is not corrupted
   - Verify file encoding (UTF-8 recommended)

3. **Model Errors**:
   - Ensure you're using a compatible version of the Groq API
   - Check your internet connection

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.