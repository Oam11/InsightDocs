import os
from typing import List, Dict, Any, Tuple, Optional
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.schema import Document
import PyPDF2
import pdfplumber
from docx import Document as DocxDocument
import pandas as pd
try:
    import magic
    MAGIC_AVAILABLE = True
except ImportError:
    MAGIC_AVAILABLE = False
    print("Warning: python-magic not available. Using file extension for type detection.")
import tiktoken
import json
import xml.etree.ElementTree as ET
import yaml
import csv
import xlrd
import openpyxl
import pptx
import html
import re
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from datetime import datetime
import base64
from io import BytesIO
import hashlib

# Image processing imports
from PIL import Image, ImageEnhance
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("Warning: EasyOCR not available. Using Tesseract only for OCR.")

try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False
    print("Warning: Pytesseract not available. OCR functionality will be limited.")

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    import numpy as np
    print("Warning: OpenCV not available. Using basic image processing only.")

# Enhanced text processing
import nltk
from rank_bm25 import BM25Okapi

class EnhancedDocumentProcessor:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Enhanced text splitter for different document types
        self.code_splitter = RecursiveCharacterTextSplitter.from_language(
            language="python", chunk_size=800, chunk_overlap=100
        )
        
        # Initialize OCR readers
        self.easyocr_reader = None
        if EASYOCR_AVAILABLE:
            try:
                self.easyocr_reader = easyocr.Reader(['en'], gpu=False)
            except Exception as e:
                print(f"Warning: Failed to initialize EasyOCR: {e}")
                self.easyocr_reader = None
            
        # Configure Tesseract for better performance (if available)
        if PYTESSERACT_AVAILABLE:
            self.tesseract_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz .,;:!?()[]{}"-'
        else:
            self.tesseract_config = None
            
        # Document storage with metadata
        self.documents: List[Document] = []
        self.qa_history = []
        self.document_metadata = {}
        
        # Supported file types
        self.supported_text_types = {
            '.pdf', '.docx', '.txt', '.csv', '.json', '.xml', 
            '.yml', '.yaml', '.xls', '.xlsx', '.pptx', '.html', '.md'
        }
        self.supported_image_types = {
            '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.gif'
        }
        
        # Download NLTK data if needed
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
        
    def add_qa_to_history(self, question: str, answer: str):
        """Add a Q&A pair to the history."""
        self.qa_history.append({
            'question': question,
            'answer': answer,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    
    def generate_qa_pdf(self, output_path: str, qa_history: List[Dict] = None):
        """Generate a PDF containing the Q&A history."""
        # Use provided history or fall back to internal history
        history_to_use = qa_history if qa_history is not None else self.qa_history
        
        if not history_to_use:
            # Create a simple message if no Q&A history exists
            doc = SimpleDocTemplate(output_path, pagesize=letter)
            styles = getSampleStyleSheet()
            elements = []
            
            elements.append(Paragraph("Document Q&A Session", styles['Title']))
            elements.append(Spacer(1, 20))
            elements.append(Paragraph("No questions and answers recorded in this session.", styles['Normal']))
            elements.append(Spacer(1, 20))
            elements.append(Paragraph(f"Session Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
            
            doc.build(elements)
            return output_path
        
        doc = SimpleDocTemplate(output_path, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.darkblue
        )
        elements.append(Paragraph("📚 InsightDocs Q&A Session", title_style))
        
        # Session info
        session_style = ParagraphStyle(
            'SessionInfo',
            parent=styles['Normal'],
            fontSize=12,
            textColor=colors.gray
        )
        elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", session_style))
        elements.append(Paragraph(f"Total Questions: {len(history_to_use)}", session_style))
        elements.append(Spacer(1, 30))
        
        # Q&A pairs
        for i, qa in enumerate(history_to_use, 1):
            # Question
            question_style = ParagraphStyle(
                'Question',
                parent=styles['Heading2'],
                fontSize=14,
                textColor=colors.darkblue,
                spaceAfter=10,
                leftIndent=0
            )
            elements.append(Paragraph(f"Question {i}: {qa['question']}", question_style))
            
            # Answer
            answer_style = ParagraphStyle(
                'Answer',
                parent=styles['Normal'],
                fontSize=11,
                spaceAfter=15,
                leftIndent=20,
                textColor=colors.black
            )
            # Clean up the answer text for PDF
            answer_text = qa['answer'].replace('\n', '<br/>')
            elements.append(Paragraph(f"Answer: {answer_text}", answer_style))
            
            # Timestamp
            timestamp_style = ParagraphStyle(
                'Timestamp',
                parent=styles['Italic'],
                fontSize=9,
                textColor=colors.gray,
                leftIndent=20
            )
            elements.append(Paragraph(f"Asked at: {qa['timestamp']}", timestamp_style))
            elements.append(Spacer(1, 25))
        
        # Footer
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.gray,
            alignment=1  # Center alignment
        )
        elements.append(Spacer(1, 30))
        elements.append(Paragraph("Generated by InsightDocs - Multi-Format RAG Application", footer_style))
        
        # Build PDF
        doc.build(elements)
        return output_path
    
    def read_file(self, file_path: str) -> str:
        """Read different file types and return their content as text."""
        file_extension = os.path.splitext(file_path)[1].lower()
        
        # Check if it's an image file first
        if file_extension in self.supported_image_types:
            return self._read_image(file_path)
        
        # Use python-magic for better file type detection
        if MAGIC_AVAILABLE:
            try:
                mime = magic.Magic(mime=True)
                file_type = mime.from_file(file_path)
            except Exception:
                # Fallback to extension-based detection
                file_type = self._get_mime_from_extension(file_extension)
        else:
            # Use extension-based detection when magic is not available
            file_type = self._get_mime_from_extension(file_extension)
        
        # If we get a generic type, try to determine by extension
        if file_type == 'application/octet-stream' or file_type == 'application/octet-stream':
            file_type = self._get_mime_from_extension(file_extension)
        
        print(f"Processing file: {os.path.basename(file_path)} (extension: {file_extension}, mime: {file_type})")
        
        # Handle different file types
        try:
            if file_type == 'application/pdf' or file_extension == '.pdf':
                return self._read_pdf(file_path)
            elif file_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' or file_extension == '.docx':
                return self._read_docx(file_path)
            elif file_type == 'text/plain' or file_extension in ['.txt', '.md']:
                return self._read_txt(file_path)
            elif file_type == 'text/csv' or file_extension == '.csv':
                return self._read_csv(file_path)
            elif file_type == 'application/json' or file_extension == '.json':
                return self._read_json(file_path)
            elif file_type == 'text/xml' or file_extension == '.xml':
                return self._read_xml(file_path)
            elif file_type == 'text/yaml' or file_extension in ['.yml', '.yaml']:
                return self._read_yaml(file_path)
            elif file_type == 'application/vnd.ms-excel' or file_extension == '.xls':
                return self._read_xls(file_path)
            elif file_type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' or file_extension == '.xlsx':
                return self._read_xlsx(file_path)
            elif file_type == 'application/vnd.openxmlformats-officedocument.presentationml.presentation' or file_extension == '.pptx':
                return self._read_pptx(file_path)
            elif file_type == 'text/html' or file_extension == '.html':
                return self._read_html(file_path)
            else:
                # Try to read as text if file type is unknown but extension suggests text
                if file_extension in ['.txt', '.md', '.log', '.ini', '.cfg', '.conf']:
                    return self._read_txt(file_path)
                else:
                    raise ValueError(f"Unsupported file type: {file_type} for file: {os.path.basename(file_path)}")
        except Exception as e:
            # If specific reader fails, try reading as text as last resort
            if "Unsupported file type" not in str(e):
                try:
                    print(f"Primary reader failed for {os.path.basename(file_path)}, trying as text...")
                    return self._read_txt(file_path)
                except:
                    pass
            raise e
    
    def _get_mime_from_extension(self, extension: str) -> str:
        """Get MIME type from file extension as fallback."""
        mime_map = {
            '.pdf': 'application/pdf',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.txt': 'text/plain',
            '.csv': 'text/csv',
            '.json': 'application/json',
            '.xml': 'text/xml',
            '.yml': 'text/yaml',
            '.yaml': 'text/yaml',
            '.xls': 'application/vnd.ms-excel',
            '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
            '.html': 'text/html',
            '.md': 'text/plain'
        }
        return mime_map.get(extension, 'application/octet-stream')
    
    def _read_pdf(self, file_path: str) -> str:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text
    
    def _read_docx(self, file_path: str) -> str:
        doc = Document(file_path)
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])
    
    def _read_txt(self, file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    def _read_csv(self, file_path: str) -> str:
        df = pd.read_csv(file_path)
        return df.to_string()
    
    def _read_json(self, file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            return json.dumps(data, indent=2)
    
    def _read_xml(self, file_path: str) -> str:
        tree = ET.parse(file_path)
        root = tree.getroot()
        return ET.tostring(root, encoding='unicode', method='xml')
    
    def _read_yaml(self, file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
            return yaml.dump(data, default_flow_style=False)
    
    def _read_xls(self, file_path: str) -> str:
        workbook = xlrd.open_workbook(file_path)
        text = ""
        for sheet in workbook.sheets():
            text += f"Sheet: {sheet.name}\n"
            for row in range(sheet.nrows):
                text += "\t".join(str(sheet.cell_value(row, col)) for col in range(sheet.ncols)) + "\n"
        return text
    
    def _read_xlsx(self, file_path: str) -> str:
        workbook = openpyxl.load_workbook(file_path, data_only=True)
        text = ""
        for sheet in workbook.sheetnames:
            text += f"Sheet: {sheet}\n"
            ws = workbook[sheet]
            for row in ws.rows:
                text += "\t".join(str(cell.value) for cell in row) + "\n"
        return text
    
    def _read_pptx(self, file_path: str) -> str:
        prs = pptx.Presentation(file_path)
        text = ""
        for slide in prs.slides:
            text += f"Slide {slide.slide_number}\n"
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text += shape.text + "\n"
        return text
    
    def _read_html(self, file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as file:
            html_content = file.read()
            # Remove HTML tags
            text = re.sub('<[^<]+?>', '', html_content)
            # Decode HTML entities
            text = html.unescape(text)
            return text
    
    def process_documents(self, file_paths: List[str], file_mapping: Dict[str, str] = None) -> Dict[str, Any]:
        """Process multiple documents and create enhanced vector stores."""
        all_chunks = []
        all_documents = []
        file_types = {}
        processing_errors = []
        
        for file_path in file_paths:
            # Get the original filename if mapping is provided
            if file_mapping and file_path in file_mapping:
                original_name = file_mapping[file_path]
                display_name = original_name
            else:
                original_name = os.path.basename(file_path)
                display_name = original_name
            
            try:
                # Read the file content
                content = self.read_file(file_path)
                
                if not content or not content.strip():
                    print(f"Warning: No content extracted from {display_name}")
                    processing_errors.append(f"{display_name}: No readable content found")
                    continue
                
                # Split content into chunks
                file_extension = os.path.splitext(file_path)[1].lower()
                if file_extension in ['.py', '.js', '.ts', '.java', '.cpp', '.c']:
                    chunks = self.code_splitter.split_text(content)
                else:
                    chunks = self.text_splitter.split_text(content)
                
                if not chunks:
                    print(f"Warning: No chunks created from {display_name}")
                    processing_errors.append(f"{display_name}: Could not create text chunks")
                    continue
                
                # Add file metadata to each chunk
                content_type = self._get_content_type(file_extension)
                file_types[display_name] = content_type
                
                # Create documents with metadata
                for i, chunk in enumerate(chunks):
                    if chunk.strip():  # Only add non-empty chunks
                        doc_content = f"[Document: {display_name} | Type: {content_type} | Chunk: {i+1}]\n{chunk}"
                        doc = Document(
                            page_content=doc_content,
                            metadata={
                                "source": display_name,
                                "file_path": file_path,
                                "original_name": original_name,
                                "content_type": content_type,
                                "chunk_id": i+1,
                                "file_extension": file_extension
                            }
                        )
                        all_documents.append(doc)
                        all_chunks.append(doc_content)
                
                print(f"✅ Processed {display_name}: {len([c for c in chunks if c.strip()])} chunks")
                
            except Exception as e:
                error_msg = f"{display_name}: {str(e)}"
                print(f"❌ Error processing {error_msg}")
                processing_errors.append(error_msg)
                continue
        
        # Provide detailed error information
        if not all_chunks:
            error_details = "\n".join(processing_errors) if processing_errors else "Unknown processing error"
            raise ValueError(f"""No text content could be extracted from any of the uploaded files. 

Processing Details:
{error_details}

Troubleshooting Tips:
1. Ensure files are not corrupted or password-protected
2. Check that file extensions match their actual content
3. For images, ensure they contain readable text
4. Try uploading files individually to identify problematic ones
5. Supported formats: PDF, Word, Excel, PowerPoint, Text, Images, etc.""")
        
        # Show processing summary
        if processing_errors:
            print(f"\n⚠️ Processing Summary:")
            print(f"✅ Successfully processed: {len(file_types)} files")
            print(f"❌ Failed to process: {len(processing_errors)} files")
            for error in processing_errors:
                print(f"   - {error}")
        
        # Create FAISS vector store
        vector_store = FAISS.from_documents(all_documents, self.embeddings)
        
        # Create BM25 retriever for keyword-based search
        bm25_retriever = BM25Retriever.from_documents(all_documents)
        bm25_retriever.k = 5
        
        # Store for later use in ensemble retrieval
        self.documents = all_documents
        self.document_metadata = {
            'file_types': file_types,
            'total_chunks': len(all_chunks),
            'total_files': len(file_types),
            'processing_errors': processing_errors
        }
        
        return {
            'vector_store': vector_store,
            'bm25_retriever': bm25_retriever,
            'documents': all_documents,
            'metadata': self.document_metadata
        }
    
    def _get_content_type(self, file_extension: str) -> str:
        """Get a human-readable content type based on file extension."""
        content_types = {
            '.pdf': 'PDF Document',
            '.docx': 'Word Document',
            '.txt': 'Text File',
            '.csv': 'CSV Spreadsheet',
            '.json': 'JSON Data',
            '.xml': 'XML Document',
            '.yml': 'YAML File',
            '.yaml': 'YAML File',
            '.xls': 'Excel Spreadsheet',
            '.xlsx': 'Excel Spreadsheet',
            '.pptx': 'PowerPoint Presentation',
            '.html': 'HTML Document',
            '.md': 'Markdown Document',
            '.jpg': 'JPEG Image',
            '.jpeg': 'JPEG Image',
            '.png': 'PNG Image',
            '.bmp': 'BMP Image',
            '.tiff': 'TIFF Image',
            '.tif': 'TIFF Image',
            '.webp': 'WebP Image',
            '.gif': 'GIF Image',
            '.py': 'Python Code',
            '.js': 'JavaScript Code',
            '.ts': 'TypeScript Code',
            '.java': 'Java Code',
            '.cpp': 'C++ Code',
            '.c': 'C Code'
        }
        return content_types.get(file_extension, 'Unknown Type')
    
    def create_qa_chain(self, document_store: Dict[str, Any]) -> RetrievalQA:
        """Create an enhanced question-answering chain using ensemble retrieval."""
        llm = ChatGroq(
            groq_api_key=self.api_key,
            model_name="gemma2-9b-it",
            temperature=0.3,  # Lower temperature for more focused answers
            max_tokens=4096
        )
        
        # Get retrievers from the document store
        vector_store = document_store['vector_store']
        bm25_retriever = document_store['bm25_retriever']
        
        # Create ensemble retriever combining semantic and keyword search
        vector_retriever = vector_store.as_retriever(
            search_kwargs={
                "k": 8,  # Increased for better context
                "fetch_k": 20,
                "lambda_mult": 0.6  # Balance between relevance and diversity
            }
        )
        
        # Combine retrievers with different weights
        ensemble_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[0.6, 0.4]  # Favor semantic search slightly
        )
        
        # Create QA chain with custom prompt
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=ensemble_retriever,
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": self._create_custom_prompt()
            }
        )
        return qa_chain
    
    def _create_custom_prompt(self):
        """Create a custom prompt template for better responses."""
        from langchain.prompts import PromptTemplate
        
        template = """Use the following pieces of context to answer the question at the end. 
        The context may include text from various document types including PDFs, Word documents, spreadsheets, presentations, images with OCR text, and other file formats.
        
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        When answering, consider the document type and source of the information.
        
        Context:
        {context}
        
        Question: {question}
        
        Helpful Answer:"""
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocess image for better OCR accuracy."""
        if not CV2_AVAILABLE:
            # Fallback to basic PIL processing
            pil_img = Image.open(image_path)
            # Convert to grayscale and return as numpy array
            gray_img = pil_img.convert('L')
            return np.array(gray_img)
        
        # Read image with OpenCV
        img = cv2.imread(image_path)
        if img is None:
            # Try with PIL if OpenCV fails
            pil_img = Image.open(image_path)
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply denoising
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Morphological operations to clean up
        kernel = np.ones((1, 1), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def extract_text_from_image(self, image_path: str) -> str:
        """Extract text from image using available OCR methods."""
        extracted_texts = []
        
        # Method 1: EasyOCR (if available)
        if EASYOCR_AVAILABLE and self.easyocr_reader:
            try:
                results = self.easyocr_reader.readtext(image_path)
                easyocr_text = " ".join([result[1] for result in results if result[2] > 0.5])
                if easyocr_text.strip():
                    extracted_texts.append(("EasyOCR", easyocr_text))
            except Exception as e:
                print(f"EasyOCR failed: {e}")
        
        # Method 2: Tesseract with preprocessing (if available)
        if PYTESSERACT_AVAILABLE and CV2_AVAILABLE:
            try:
                # Preprocess image
                processed_img = self.preprocess_image(image_path)
                
                # Convert back to PIL format for Tesseract
                pil_img = Image.fromarray(processed_img)
                
                # Extract text with Tesseract
                tesseract_text = pytesseract.image_to_string(pil_img, config=self.tesseract_config)
                if tesseract_text.strip():
                    extracted_texts.append(("Tesseract_Processed", tesseract_text))
            except Exception as e:
                print(f"Tesseract with preprocessing failed: {e}")
        
        # Method 3: Tesseract on original image (fallback)
        if PYTESSERACT_AVAILABLE:
            try:
                original_text = pytesseract.image_to_string(Image.open(image_path))
                if original_text.strip():
                    extracted_texts.append(("Tesseract_Original", original_text))
            except Exception as e:
                print(f"Tesseract on original failed: {e}")
        
        # Combine results and choose the best one
        if not extracted_texts:
            return "No text could be extracted from this image. OCR libraries may not be available."
        
        # Return the longest text as it's likely more complete
        best_text = max(extracted_texts, key=lambda x: len(x[1]))
        return best_text[1]
    
    def analyze_image_metadata(self, image_path: str) -> Dict[str, Any]:
        """Extract metadata from image."""
        try:
            with Image.open(image_path) as img:
                metadata = {
                    'format': img.format,
                    'mode': img.mode,
                    'size': img.size,
                    'filename': os.path.basename(image_path)
                }
                
                # Extract EXIF data if available
                if hasattr(img, '_getexif') and img._getexif():
                    metadata['exif'] = img._getexif()
                
                return metadata
        except Exception:
            return {'filename': os.path.basename(image_path)}
    
    def _read_image(self, file_path: str) -> str:
        """Process image file and extract text content."""
        try:
            # Extract text using OCR
            text_content = self.extract_text_from_image(file_path)
            
            # Get image metadata
            metadata = self.analyze_image_metadata(file_path)
            
            # Create formatted content
            content = f"""Image Analysis Report
Filename: {metadata.get('filename', 'Unknown')}
Format: {metadata.get('format', 'Unknown')}
Size: {metadata.get('size', 'Unknown')}
Mode: {metadata.get('mode', 'Unknown')}

Extracted Text Content:
{text_content}

---
"""
            
            return content
        except Exception as e:
            return f"Error processing image {os.path.basename(file_path)}: {str(e)}"

# Backward compatibility alias
DocumentProcessor = EnhancedDocumentProcessor