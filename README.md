# 📚 InsightDocs - Multi-Format RAG Application

A powerful Retrieval-Augmented Generation (RAG) application that processes multiple document types including images using advanced OCR technology. Chat with your documents and images using natural language queries with zero configuration required.

## ✨ Features

### 🔑 **Zero Configuration Setup**
- **No Config Files Required**: Enter your API key directly in the web interface
- **Local Storage**: API key is saved locally so you only enter it once
- **Session-Based Security**: Option to use session-only storage for enhanced privacy
- **Instant Setup**: Get running in under 2 minutes

### 📄 **Comprehensive Document Support**
- **Text Documents**: PDF, Word (.docx), Text files (.txt), Markdown (.md)
- **Spreadsheets**: Excel (.xlsx, .xls), CSV files
- **Presentations**: PowerPoint (.pptx)
- **Data Files**: JSON, XML, YAML
- **Web Content**: HTML documents
- **Images with OCR**: JPG, PNG, TIFF, BMP, WebP, GIF

### 🧠 **Advanced RAG Technology**
- **Hybrid Retrieval**: Combines semantic search (FAISS) with keyword search (BM25)
- **Ensemble Retrieval**: Intelligently weights different search methods (60% semantic + 40% keyword)
- **Dual OCR Engines**: EasyOCR + Tesseract for maximum text extraction accuracy
- **Smart Chunking**: Document-type aware text splitting for optimal context
- **Source Attribution**: Shows exactly which documents provided each answer
- **Metadata Preservation**: Tracks file types, sources, and chunk information

### 🎯 **User Experience**
- **Real-time Processing Stats**: See files processed, chunks created, and error details
- **Source References**: Expandable sections showing document sources for each answer
- **Error Handling**: Detailed feedback when files can't be processed
- **PDF Export**: Download complete Q&A sessions as formatted reports
- **Progress Tracking**: Visual indicators and processing summaries

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- A free Groq API key from [console.groq.com](https://console.groq.com)

### Quick Install

1. **Clone the repository:**
```bash
git clone https://github.com/Oam11/InsightDocs.git
cd InsightDocs
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Optional - Install Tesseract OCR (for better image processing):**
   - **Windows**: Download from [GitHub Tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
   - **macOS**: `brew install tesseract`
   - **Linux**: `sudo apt-get install tesseract-ocr`

4. **Run the application:**
```bash
streamlit run app.py
```

5. **Open your browser** and go to `http://localhost:8501`

6. **Enter your Groq API key** when prompted (get one free at [console.groq.com](https://console.groq.com))
   - Check "Remember this key" to store it locally for future use
   - Your API key will be saved in `~/.insightdocs/config.json` for convenience

That's it! No configuration files needed.

## 🎮 How to Use

### 1. **Get Your API Key**
- Visit [console.groq.com](https://console.groq.com) and create a free account
- Generate an API key (starts with `gsk_`)
- Enter it in the app when prompted

### 2. **Upload Your Documents**
- Drag and drop files or use the file browser
- Mix different types: PDFs, images, spreadsheets, presentations
- Upload multiple files at once for comprehensive analysis

### 3. **Process Documents**
- Click "Process Documents" 
- View the processing summary showing:
  - Total files processed
  - Number of text chunks created
  - File types detected
  - Any processing errors

### 4. **Ask Questions**
- Use natural language queries
- Be specific for better results
- Reference document types when needed

### 5. **Review Answers**
- Get detailed responses with source attribution
- Click "Source Documents Used" to see which files provided the answer
- Download your Q&A session as a PDF

## 💡 Example Use Cases & Questions

### **Business Intelligence**
```
"What are the key performance metrics mentioned in the quarterly report?"
"List all action items from the meeting minutes"
"Compare sales figures between Q1 and Q2"
```

### **Research & Analysis**
```
"Summarize the main findings from all research papers"
"What methodologies were used in the studies?"
"Extract all statistical data mentioned in the documents"
```

### **Document Review**
```
"Find all references to budget allocations"
"What are the compliance requirements mentioned?"
"List all contact information from the uploaded files"
```

### **Image Analysis**
```
"What text is visible in the uploaded screenshots?"
"Extract data from the chart in the image"
"What information can you read from the scanned document?"
```

## 🔧 Troubleshooting

### **"No text content could be extracted" Error**
This usually happens when files can't be read properly:

1. **Test with simple file**: Try uploading the included `test_document.txt`
2. **Check file formats**: Ensure files have proper extensions
3. **Remove passwords**: Documents must not be password-protected
4. **Upload individually**: Test files one at a time to identify issues
5. **Check file corruption**: Open files in their native applications first

### **API Key Issues**
- Ensure your key starts with `gsk_`
- Verify your Groq account is active
- Try generating a new API key if issues persist

### **OCR Not Working**
- Install Tesseract OCR for better image processing
- Use high-resolution, clear images
- Ensure good contrast between text and background

### **Upload Issues**
- Check file size limits (default 200MB per file)
- Try smaller files first
- Use supported file formats only

## 📊 Technical Architecture

### **Core Components**
- **Frontend**: Streamlit with enhanced UI/UX
- **Embeddings**: SentenceTransformers all-MiniLM-L6-v2 (CPU optimized)
- **Vector Store**: FAISS for semantic search
- **Keyword Search**: BM25 for exact term matching
- **OCR**: Dual-engine approach (EasyOCR + Tesseract)

### **Performance Features**
- **Lightweight**: No heavy transformers, CPU optimized
- **Fast Processing**: Efficient document chunking and indexing
- **Hybrid Search**: Best of both semantic and keyword retrieval
- **Memory Efficient**: Optimized for standard hardware

### **File Processing Pipeline**
1. **Upload & Validation**: Files saved with proper extensions
2. **Type Detection**: Smart MIME type and extension-based detection
3. **Content Extraction**: Format-specific readers with fallbacks
4. **OCR Processing**: Images processed with dual OCR engines
5. **Smart Chunking**: Context-aware text splitting
6. **Dual Indexing**: Both vector and keyword indexes created
7. **Ensemble Retrieval**: Weighted combination of search methods

## 📈 Supported File Formats

| Category | Formats | Notes |
|----------|---------|-------|
| **Documents** | PDF, DOCX, TXT, MD | Text-based content only |
| **Spreadsheets** | XLSX, XLS, CSV | All sheets processed |
| **Presentations** | PPTX | Text and slide content |
| **Images** | JPG, PNG, TIFF, BMP, WebP, GIF | OCR text extraction |
| **Data** | JSON, XML, YAML | Structured data parsing |
| **Web** | HTML | Text content extraction |
| **Code** | PY, JS, TS, JAVA, CPP, C | Code-aware chunking |

## 🏆 Key Advantages

### **vs. Traditional RAG Systems**
- ✅ **Zero Configuration**: No complex setup or config files
- ✅ **Multi-Modal**: Handles both text and images seamlessly
- ✅ **Hybrid Search**: Better retrieval than pure vector search
- ✅ **Source Attribution**: Always know where answers come from
- ✅ **Error Resilience**: Graceful handling of problematic files

### **vs. Chat Interfaces**
- ✅ **Document Context**: Maintains awareness of document structure
- ✅ **Batch Processing**: Handle multiple documents simultaneously
- ✅ **Persistent Sessions**: Keep context across conversations
- ✅ **Export Capability**: Generate reports from your analysis

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **[Groq](https://groq.com/)** - Ultra-fast LLM inference
- **[Streamlit](https://streamlit.io/)** - Elegant web framework
- **[LangChain](https://langchain.com/)** - RAG orchestration
- **[EasyOCR](https://github.com/JaidedAI/EasyOCR)** - OCR capabilities
- **[Tesseract](https://github.com/tesseract-ocr/tesseract)** - Text recognition

---

**🚀 Ready to explore your documents?**  
Run `streamlit run app.py` and start chatting with your files in minutes!

### 🎯 Core Capabilities
- **Multi-Document & Image Support**: Upload and process multiple files simultaneously
- **Advanced OCR**: Extract text from images using EasyOCR and Tesseract
- **Hybrid Retrieval**: Combines semantic search (FAISS) with keyword search (BM25)
- **Session Management**: Unique session tracking with persistent chat history
- **PDF Export**: Download your entire Q&A session as a formatted PDF

### 📄 Supported File Formats
**Documents:**
- PDF documents
- Word documents (DOCX)
- Text files (TXT, MD)
- Spreadsheets (CSV, XLS, XLSX)
- JSON & XML data
- YAML configuration files
- PowerPoint presentations (PPTX)
- HTML documents

**Images (with OCR):**
- JPEG, PNG, TIFF, BMP
- WebP, GIF formats
- Automatic text extraction
- Image preprocessing for better OCR accuracy

**Code Files:**
- Python, JavaScript, TypeScript
- Java, C++, C
- Optimized chunking for code structure

### 🧠 Enhanced RAG Features
- **Ensemble Retrieval**: Combines vector similarity and keyword matching
- **Smart Chunking**: Document-type aware text splitting
- **Source Tracking**: Know which documents provided each answer
- **Metadata Enrichment**: Rich context with file types and chunk information
- **Lightweight Architecture**: No heavy transformers, optimized for efficiency

## Prerequisites

- Python 3.8 or higher
- Groq API key

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/Oam11/InsightDocs.git
cd InsightDocs
```

2. Create and activate a virtual environment:
```bash
python -m venv vev
# On Windows:
vev\Scripts\activate
# On macOS/Linux:
source vev/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. **Set up OCR (Optional but recommended):**
   - Install Tesseract OCR: [Installation Guide](https://github.com/tesseract-ocr/tesseract)
   - On Windows: Download from [GitHub releases](https://github.com/UB-Mannheim/tesseract/wiki)
   - On macOS: `brew install tesseract`
   - On Ubuntu: `sudo apt install tesseract-ocr`

5. **Get your Groq API key:**
   - Visit [console.groq.com](https://console.groq.com)
   - Sign up for a free account
   - Generate an API key (starts with `gsk_`)
   - No configuration files needed - you'll enter it directly in the app!

## 🎮 Usage

1. **Start the application:**
```bash
streamlit run app.py
```

2. **Upload documents and images** using the sidebar file uploader
3. **Click "Process Documents"** to analyze content with OCR and text extraction
4. **Ask questions** about your uploaded content
5. **View source references** to see which documents provided the answers
6. **Download Q&A session** as a formatted PDF

## 🔧 How It Works

### 📊 Document Processing Pipeline
1. **File Type Detection**: Automatic identification of document and image types
2. **OCR Processing**: Extract text from images using dual OCR engines
3. **Smart Chunking**: Document-type aware text splitting for optimal context
4. **Hybrid Indexing**: Combines FAISS vector store with BM25 keyword search
5. **Metadata Enrichment**: Track sources, chunk IDs, and content types

### 🧠 Enhanced RAG System
1. **Ensemble Retrieval**: Combines semantic and keyword search (60/40 weight)
2. **Context-Aware Generation**: Uses retrieved chunks with source information
3. **Source Attribution**: Shows which documents contributed to each answer
4. **Session Persistence**: Maintains conversation history and document context

### 🖼️ Image Processing Features
- **Dual OCR Engines**: EasyOCR + Tesseract for maximum accuracy
- **Image Preprocessing**: Denoising, thresholding, morphological operations
- **Format Support**: JPEG, PNG, TIFF, BMP, WebP, GIF
- **Metadata Extraction**: Image properties, EXIF data when available

## 🛠️ Technical Architecture

**Core Components:**
- **Frontend**: Streamlit with enhanced UI/UX
- **Embeddings**: SentenceTransformers all-MiniLM-L6-v2 (CPU optimized)
- **Vector Store**: FAISS for semantic search
- **Keyword Search**: BM25Okapi for exact matching
- **OCR**: EasyOCR + Tesseract (dual engine approach)

**Performance Optimizations:**
- Lightweight architecture (no heavy transformers)
- CPU-optimized embeddings
- Efficient document chunking
- Ensemble retrieval for accuracy
- Minimal bandwidth requirements

## 🎯 Use Cases

- **Research**: Analyze academic papers, reports, and documents
- **Business Intelligence**: Extract insights from presentations, spreadsheets
- **Document Management**: Search through large document collections
- **Image Analysis**: Extract text from scanned documents, charts, diagrams
- **Code Review**: Understand code repositories and documentation
- **Legal/Compliance**: Review contracts, policies, and regulatory documents

## 🔍 Troubleshooting

**Common Issues:**

1. **API Key Problems**:
   - Ensure your Groq API key is correctly set in `.streamlit/secrets.toml`
   - Verify the API key format starts with `gsk_`
   - Check API key permissions and rate limits

2. **OCR Issues**:
   - Install Tesseract OCR if image processing fails
   - Ensure image files are clear and readable
   - Try different image formats (PNG often works better than JPEG)

3. **File Processing Errors**:
   - Check if the file format is supported
   - Ensure files aren't corrupted or password-protected
   - Try processing files individually to isolate issues

4. **Performance Optimization**:
   - Process documents in smaller batches for large collections
   - Use high-quality images for better OCR accuracy
   - Clear browser cache if UI becomes unresponsive

## 📈 Roadmap

- [ ] Support for more image formats (SVG, HEIC)
- [ ] Advanced document layout analysis
- [ ] Multi-language OCR support
- [ ] Integration with cloud storage (Google Drive, Dropbox)
- [ ] Real-time collaboration features
- [ ] API endpoint for programmatic access

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Groq](https://groq.com/) for the fast inference API
- [Streamlit](https://streamlit.io/) for the web framework
- [LangChain](https://langchain.com/) for RAG components
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) for OCR capabilities
- [Tesseract](https://github.com/tesseract-ocr/tesseract) for text recognition

---

**Built with ❤️ for the AI community**
   - Verify file encoding (UTF-8 recommended)

3. **Model Errors**:
   - Ensure you're using a compatible version of the Groq API
   - Check your internet connection

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.