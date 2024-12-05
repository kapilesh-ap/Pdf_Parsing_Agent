from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import os
from groq import Groq
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import PyPDF2
import io
import pickle
from datetime import datetime
from pinecone_ops import PineconeManager
from dotenv import load_dotenv
from agents import AgentSystem
import asyncio
from functools import wraps

# Load environment variables
load_dotenv()

# Initialize Flask app and CORS
app = Flask(__name__)
CORS(app)

# Helper function to run async code
def async_route(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))
    return wrapped

# Configure logging
logging.basicConfig(
    filename='app.log',
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s'
)

# Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX = "pdf-embeddings"
ALLOWED_EXTENSIONS = {'pdf'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

# Load embedding model for RAG
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize FAISS index
embedding_dim = 384  # Dimension for all-MiniLM-L6-v2
faiss_index = faiss.IndexFlatL2(embedding_dim)
sections = []  # Store text chunks and metadata

# Global PDF metadata
pdf_metadata = {}

# Initialize Pinecone manager
pinecone_manager = PineconeManager(PINECONE_API_KEY, PINECONE_ENV, PINECONE_INDEX)

def check_pinecone_connection():
    """Verify Pinecone connection and index"""
    try:
        logging.info("Checking Pinecone connection...")
        # Test vector with correct dimension
        test_vector = np.zeros(384)  # Match your embedding dimension
        test_result = pinecone_manager.query(test_vector, top_k=1)
        
        if hasattr(test_result, 'matches'):
            logging.info("✅ Pinecone connection successful")
            return True
        else:
            logging.error("❌ Pinecone connection failed - invalid response format")
            return False
            
    except Exception as e:
        logging.error(f"❌ Error connecting to Pinecone: {str(e)}")
        return False

# Initialize Pinecone connection
PINECONE_CONNECTED = check_pinecone_connection()

def allowed_file(filename):
    """Check if the file type is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Initialize Groq LLM client
groq_client = Groq(api_key=GROQ_API_KEY)
groq_llm = ChatGroq(
    temperature=0,
    model="llama-3.1-70b-versatile",
    api_key=GROQ_API_KEY
)

# Initialize the Agent System
agent_system = AgentSystem(GROQ_API_KEY)

# Add these near the top with other global variables
current_pdf_id = 0

def get_next_pdf_id():
    """Get next incremental PDF ID"""
    global current_pdf_id
    current_pdf_id += 1
    return str(current_pdf_id)

# Add function to load last PDF ID from storage
def load_last_pdf_id():
    """Load the last used PDF ID from storage"""
    try:
        with open("last_pdf_id.txt", "r") as f:
            return int(f.read().strip())
    except FileNotFoundError:
        return 0

def save_current_pdf_id():
    """Save current PDF ID to storage"""
    with open("last_pdf_id.txt", "w") as f:
        f.write(str(current_pdf_id))

# Initialize PDF ID counter from storage
current_pdf_id = load_last_pdf_id()

def get_relevant_context(query, k=3, use_pinecone=True):
    """Get relevant context using either FAISS or Pinecone"""
    try:
        logging.info(f"Getting context for query: {query}")
        query_embedding = model.encode([query])[0]
        
        if use_pinecone and PINECONE_CONNECTED:
            logging.info(f"Using Pinecone with PDF ID: {current_pdf_id}")
            # Use Pinecone for production with PDF ID filter
            results = pinecone_manager.query(
                query_embedding, 
                pdf_id=str(current_pdf_id),
                top_k=k
            )
            
            matches = results.get('matches', [])
            logging.info(f"Pinecone returned {len(matches)} matches")
            
            if matches:
                # Get text from matches
                text_content = "\n".join(match['metadata']['text'] for match in matches)
                
                # Get metadata from first match
                first_match = matches[0]['metadata']
                metadata = {
                    'pdf_id': first_match['pdf_id'],
                    'filename': first_match['file_name'],
                    'upload_time': first_match['upload_time'],
                    'total_pages': first_match['total_pages'],
                    'file_size': first_match['file_size'],
                    'author': first_match['author'],
                    'creator': first_match['creator'],
                    'producer': first_match['producer'],
                    'subject': first_match['subject'],
                    'title': first_match['title'],
                    'creation_date': first_match['creation_date']
                }
                logging.info("Successfully retrieved context and metadata")
            else:
                logging.warning("No matches found in Pinecone")
                text_content = ""
                metadata = {}
                
        else:
            logging.info("Using FAISS for local development")
            # Use FAISS for local development
            D, I = faiss_index.search(np.array([query_embedding]).astype('float32'), k)
            relevant_sections = [sections[i] for i in I[0]]
            text_content = "\n".join(section['text'] for section in relevant_sections)
            metadata = relevant_sections[0]['metadata']['pdf_metadata'] if relevant_sections else {}

        return {
            'text': text_content,
            'pdf_metadata': metadata
        }
        
    except Exception as e:
        logging.error(f"Error getting relevant context: {str(e)}")
        logging.error(f"Full error: ", exc_info=True)
        return {'text': '', 'pdf_metadata': {}}

@app.route('/chat', methods=['POST'])
@async_route
async def chat():
    try:
        data = request.json
        question = data.get('question')
        
        if not question:
            return jsonify({"error": "No question provided"}), 400
        
        if not PINECONE_CONNECTED:
            logging.error("Pinecone is not connected")
            return jsonify({"error": "Vector database not available"}), 503
            
        # Get relevant context using Pinecone
        context = get_relevant_context(question, use_pinecone=True)
        
        if not context.get('text'):
            logging.warning("No context found for the query")
        
        # Process query through agent system
        response = await agent_system.process_query(question, context)
        formatted_response = agent_system.format_response(response)
        
        # Add agent outputs to the response
        formatted_response.update({
            "agent_outputs": {
                "user_analysis": response.get("user_analysis", {}),
                "rag_prompt": response.get("rag_prompt", "")
            }
        })
        
        return jsonify(formatted_response), 200
        
    except Exception as e:
        logging.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

def process_and_store_pdf(text, file_name, pdf_reader, pdf_id):
    """Process PDF content and store in both FAISS and Pinecone"""
    try:
        global faiss_index, sections
        
        # Create PDF metadata
        pdf_metadata = {
            'pdf_id': pdf_id,  # Make sure to include pdf_id
            'filename': file_name,
            'upload_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'total_pages': len(pdf_reader.pages),
            'file_size': len(text),
            'author': pdf_reader.metadata.get('/Author', 'Unknown'),
            'creator': pdf_reader.metadata.get('/Creator', 'Unknown'),
            'producer': pdf_reader.metadata.get('/Producer', 'Unknown'),
            'subject': pdf_reader.metadata.get('/Subject', 'Unknown'),
            'title': pdf_reader.metadata.get('/Title', 'Unknown'),
            'creation_date': pdf_reader.metadata.get('/CreationDate', 'Unknown'),
        }

        # Split and embed text
        chunks = split_text(text)
        embeddings = model.encode(chunks)
        
        processed_chunks = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            processed_chunks.append({
                'text': chunk,
                'embedding': embedding,
                'page_number': i // 2
            })

        logging.info(f"Created {len(processed_chunks)} chunks for processing")

        # Store in Pinecone
        if PINECONE_CONNECTED:
            pinecone_success = pinecone_manager.upsert_document(processed_chunks, pdf_metadata)
            if not pinecone_success:
                logging.error("Failed to store document in Pinecone")
                return {"success": False, "error": "Failed to store in vector database"}
            logging.info("✅ Document stored in Pinecone")
        
        # Store locally for FAISS
        faiss_index.add(np.array([chunk['embedding'] for chunk in processed_chunks]).astype('float32'))
        for i, chunk in enumerate(processed_chunks):
            sections.append({
                'text': chunk['text'],
                'metadata': {
                    'file_name': file_name,
                    'chunk_id': len(sections) + i,
                    'source': 'pdf',
                    'pdf_metadata': pdf_metadata
                }
            })
        
        # Save FAISS index and sections
        faiss.write_index(faiss_index, "faiss.index")
        with open("sections.pkl", "wb") as f:
            pickle.dump(sections, f)

        return {
            "success": True,
            "chunks_processed": len(chunks),
            "message": "PDF processed and stored successfully",
            "metadata": pdf_metadata
        }

    except Exception as e:
        logging.error(f"Error processing PDF: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

@app.route('/upload', methods=['POST'])
def upload_file():
    if not PINECONE_CONNECTED:
        return jsonify({
            "error": "Vector database not available. Please check your configuration."
        }), 503
        
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        try:
            # Clear previous data before processing new file
            clear_previous_data()
            
            # Generate next incremental PDF ID
            global current_pdf_id
            pdf_id = get_next_pdf_id()
            save_current_pdf_id()
            
            logging.info(f"Processing PDF with ID: {pdf_id}")

            # Save file content
            file_content = file.read()
            
            # Read PDF content
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
            text_content = ""
            for page in pdf_reader.pages:
                text_content += page.extract_text()

            # Get PDF info
            info = pdf_reader.metadata
            
            # Process and store in vector database
            storage_result = process_and_store_pdf(text_content, file.filename, pdf_reader, pdf_id)
            
            if not storage_result["success"]:
                logging.error(f"Failed to store PDF: {storage_result['error']}")
                return jsonify({"error": storage_result["error"]}), 500

            logging.info("PDF processed and stored successfully")
            
            # Store PDF metadata globally
            global pdf_metadata
            pdf_metadata = {
                'pdf_id': pdf_id,
                'filename': file.filename,
                'upload_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'total_pages': len(pdf_reader.pages),
                'file_size': len(file_content),
                'author': info.get('/Author', 'Unknown'),
                'creator': info.get('/Creator', 'Unknown'),
                'producer': info.get('/Producer', 'Unknown'),
                'subject': info.get('/Subject', 'Unknown'),
                'title': info.get('/Title', 'Unknown'),
                'creation_date': info.get('/CreationDate', 'Unknown'),
            }
            
            return jsonify({
                "success": True,
                "message": "File uploaded and processed successfully",
                "pdf_id": pdf_id,
                "metadata": pdf_metadata
            }), 200

        except Exception as e:
            logging.error(f"Error processing file: {e}")
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "File type not allowed"}), 400

@app.route('/reset-agents', methods=['POST'])
def reset_agents():
    try:
        # Reset the agent system
        global agent_system
        agent_system = AgentSystem(GROQ_API_KEY)
        
        return jsonify({"success": True}), 200
    except Exception as e:
        logging.error(f"Error resetting agents: {str(e)}")
        return jsonify({"error": "Failed to reset agents"}), 500

def split_text(text, chunk_size=1000, overlap=200):
    """Split text into overlapping chunks"""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        if end > text_length:
            end = text_length
        
        chunk = text[start:end]
        chunks.append(chunk)
        
        start = end - overlap
        
        if start + chunk_size > text_length:
            if start < text_length:
                chunks.append(text[start:])
            break
    
    return chunks

def clear_previous_data():
    """Clear previous PDF data from memory"""
    global faiss_index, sections
    faiss_index = faiss.IndexFlatL2(embedding_dim)  # Reset FAISS index
    sections = []  # Clear sections

def parse_pdf_content(text):
    """Analyze PDF content"""
    prompt = f"""
    Analyze the following PDF content and provide a detailed summary. 
    Include key points, main ideas, and important details from the text.
    Format the response in a clear, structured way.

    PDF Content:
    {text}

    Provide a comprehensive analysis of the content above.
    """
    try:
        response = groq_llm.invoke(prompt)
        return {"analysis": response.content}
    except Exception as e:
        logging.error(f"Error analyzing PDF with LLM: {str(e)}")
        return {"error": f"Error analyzing PDF: {str(e)}"}

if __name__ == '__main__':
    # Load existing index and sections if they exist
    if os.path.exists("faiss.index") and os.path.exists("sections.pkl"):
        faiss_index = faiss.read_index("faiss.index")
        with open("sections.pkl", "rb") as f:
            sections = pickle.load(f)
    
    app.run(debug=True)