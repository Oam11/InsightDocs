# Document Chat with Gemini

A Streamlit application that allows you to chat with your documents using Google's Gemini AI. This application implements a RAG (Retrieval Augmented Generation) model to provide accurate answers based on your document content.

## Features

- Support for multiple file types (PDF, DOCX, TXT, CSV, JSON, XML, YAML, Excel, PowerPoint, HTML)
- Batch processing of documents
- Token limit management
- Interactive chat interface
- Secure API key handling using environment variables
- Automatic file type detection
- Efficient text chunking and retrieval

## Prerequisites

- Python 3.8 or higher
- Google Gemini API key

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/Chat-with-documents.git
cd Chat-with-documents
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root directory and add your Gemini API key:
```bash
GEMINI_API_KEY=your_api_key_here
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

3. Upload your documents using the file uploader in the sidebar

4. Click "Process Documents" to analyze your documents

5. Start asking questions about your documents in the chat interface

## Supported File Types

### Text Documents
- Text files (.txt)
- PDF files (.pdf)
- Microsoft Word documents (.docx)
- HTML files (.html)

### Spreadsheets
- CSV files (.csv)
- Excel files (.xls, .xlsx)

### Data Files
- JSON files (.json)
- XML files (.xml)
- YAML files (.yml, .yaml)

### Presentations
- PowerPoint files (.pptx)

### Additional Features
- Automatic file type detection using MIME types
- Fallback to text reading for unknown file types
- Structured data extraction from various formats
- Preserved formatting where applicable

## How It Works

1. **Document Processing**:
   - The application processes your documents and splits them into manageable chunks
   - Each file type is handled with its specific parser to extract text content
   - Documents are processed in batches to manage token limits

2. **Embedding Generation**:
   - Creates embeddings for each chunk using Google's Generative AI
   - Uses the embedding-001 model for efficient vector representation

3. **Vector Storage**:
   - The chunks are stored in a FAISS vector database for efficient retrieval
   - FAISS enables fast similarity search across document chunks

4. **Question Answering**:
   - When you ask a question, the system:
     - Retrieves the most relevant document chunks
     - Uses Gemini AI to generate a response based on the retrieved context
   - The RAG model ensures responses are grounded in your document content

## Security

- API key is stored securely in environment variables
- All processing is done locally
- Documents are processed in temporary storage
- The `.env` file should be added to `.gitignore` to prevent accidental commits
- No document content is stored permanently

## Performance Considerations

- Documents are processed in chunks to manage token limits
- Text splitting is optimized for context preservation
- Vector search is efficient using FAISS
- Batch processing for multiple documents

## Troubleshooting

If you encounter any issues:

1. **API Key Issues**:
   - Ensure your API key is correctly set in the `.env` file
   - Verify the API key has the necessary permissions

2. **File Processing Issues**:
   - Check if the file type is supported
   - Ensure the file is not corrupted
   - Verify file encoding (UTF-8 recommended)

3. **Model Errors**:
   - Ensure you're using a compatible version of the Gemini API
   - Check your internet connection

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.