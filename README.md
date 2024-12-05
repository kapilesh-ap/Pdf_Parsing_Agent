# 🤖 AI-Powered PDF Chat Assistant

An intelligent chatbot that enables natural conversations about PDF documents using advanced RAG (Retrieval Augmented Generation) and agent-based architecture.

## 🎯 Deliverables

### 1. 🧠 Intelligent Agents
- **User Agent (Using GroqLLM)**
  - 🔍 Query analysis and intent detection
  - 💬 Conversation style adaptation
  - 🧩 Context management
  - 📝 Structured query formatting

- **RAG Agent (Using GroqLLM)**
  - 🎯 Context-aware response generation
  - 📚 Document-grounded answers
  - 🔄 Fallback handling
  - ✨ Natural language synthesis

### 2. 🎨 Frontend Interface
- 💻 Modern, responsive chat interface
- 📄 PDF drag-and-drop upload
- ⚡ Real-time conversation updates
- 🐛 Debug mode for agent insights
- 🔄 Loading states and error handling

### 3. ⚙️ Backend API
- 📤 `/upload` - PDF processing endpoint
- 💭 `/chat` - Conversation endpoint
- 🔍 Vector similarity search integration
- ⚡ Async request handling

### 4. 💾 Data Management
- 🔮 Pinecone vector database integration
- 🗄️ FAISS local indexing
- 📑 PDF text chunking and embedding
- 📋 Metadata extraction and storage

## ✨ Features

### 1. 📑 PDF Processing & Management
- **PDF Upload**: 📤 Drag-and-drop interface for easy PDF uploads
- **Text Extraction**: 📝 Efficiently extracts and processes text from PDFs
- **Metadata**: 📋 Captures document properties like title, author, creation date
- **Chunking**: 🔄 Smart text segmentation for optimal processing

### 2. 🔍 Intelligent Search & Retrieval
- **Vector DB**: 🔮 Uses Pinecone for efficient similarity search
- **Local Index**: 🗄️ Maintains a FAISS index for development and testing
- **Smart Search**: 🎯 Finds the most relevant document sections for each query

### 3. 💬 Conversational Features
- **NLU**: 🧠 Processes queries in conversational language
- **Context**: 🔄 Maintains conversation history
- **Adaptation**: ✨ Matches user's communication style
- **Integration**: 🔗 Seamlessly includes document information when relevant

## 🛠️ Technology Stack

### Backend
- **Framework**: ⚡ Flask
- **AI/ML**: 
  - 🤖 Groq LLM for text generation
  - 🧮 Sentence Transformers for embeddings
  - 🔍 FAISS for local vector search
- **Vector DB**: 🔮 Pinecone
- **PDF Processing**: 📑 PyPDF2

### Frontend
- **Framework**: ⚛️ React
- **UI Components**: 🎨 Material-UI
- **Styling**: 💅 CSS with modern features
- **File Handling**: 📤 React-Dropzone

## 🚀 Installation

### Prerequisites
- 🐍 Python 3.8+
- 📦 Node.js 14+
- 🔑 Groq API key
- 🔑 Pinecone API key and environment

## 📁 Project Structure
├── backend/
│ ├── app.py # Main Flask application
│ ├── agents.py # Agent implementation
│ ├── pinecone_ops.py # Vector DB operations
│ └── requirements.txt # Python dependencies
│
└── frontend/
├── src/
│ ├── App.js # Main React component
│ ├── Chatbot.js # Chat interface
│ └── FileUpload.js# File upload component
└── package.json # Node.js dependencies

## Acknowledgments
- Groq LLM for text generation
- Pinecone for vector database
- Sentence Transformers for embeddings
- Material-UI for components