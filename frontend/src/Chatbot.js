import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { CircularProgress, IconButton, Chip } from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import InfoIcon from '@mui/icons-material/Info';
import './Chatbot.css';

function Chatbot() {
  const [input, setInput] = useState('');
  const [chatHistory, setChatHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);
  const [showAgentOutputs, setShowAgentOutputs] = useState(false);
  const [showClearButton, setShowClearButton] = useState(false);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [chatHistory]);

  const handleInputChange = (e) => {
    setInput(e.target.value);
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const formatMetadata = (metadata) => {
    if (!metadata) return null;
    return (
      <div className="metadata-section">
        <Chip 
          icon={<InfoIcon />} 
          label="Document Info" 
          className="metadata-chip"
        />
        <div className="metadata-grid">
          {Object.entries(metadata).map(([key, value]) => (
            <div key={key} className="metadata-item">
              <span className="metadata-key">{key}:</span>
              <span className="metadata-value">{value}</span>
            </div>
          ))}
        </div>
      </div>
    );
  };

  const formatAnswer = (text) => {
    // Simply return the text as a paragraph
    return <p className="chat-text">{text}</p>;
  };

  const shouldShowMetadata = (text, metadata) => {
    // Check if the question is asking about metadata
    const metadataKeywords = [
      'metadata',
      'document info',
      'file info',
      'document details',
      'file details',
      'document properties',
      'file properties',
      'tell me about the document',
      'tell me about the file'
    ];
    
    // Convert text to lowercase for case-insensitive comparison
    const lowerText = text.toLowerCase();
    return metadataKeywords.some(keyword => lowerText.includes(keyword));
  };

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMessage = { sender: 'user', text: input };
    setChatHistory(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      const response = await fetch('http://localhost:5000/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: input }),
      });
      
      const data = await response.json();
      
      if (!data.success) {
        throw new Error(data.answer);
      }

      const botMessage = { 
        sender: 'bot', 
        text: data.answer,
        metadata: data.metadata,
        showMetadata: shouldShowMetadata(input, data.metadata),
        context: data.context,
        agentOutputs: data.agent_outputs
      };
      setChatHistory(prev => [...prev, botMessage]);
    } catch (error) {
      console.error('Error:', error);
      const errorMessage = { 
        sender: 'bot', 
        text: error.message || 'Sorry, I encountered an error. Please try again.',
        isError: true
      };
      setChatHistory(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const handleClearChat = async () => {
    try {
      // Clear frontend chat history
      setChatHistory([]);
      
      // Call backend to reset agent states
      await fetch('http://localhost:5000/reset-agents', {
        method: 'POST',
      });
      
    } catch (error) {
      console.error('Error clearing chat:', error);
    }
  };

  useEffect(() => {
    setShowClearButton(chatHistory.length > 0);
  }, [chatHistory]);

  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="chatbot-container"
    >
      <div className="chat-header">
        <h2>AI Assistant</h2>
        {showClearButton && (
          <button 
            className="clear-chat-button"
            onClick={handleClearChat}
          >
            Clear Chat
          </button>
        )}
      </div>
      
      <div className="chat-history">
        <AnimatePresence>
          {chatHistory.map((msg, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, x: msg.sender === 'user' ? 20 : -20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.3 }}
              className={`chat-message ${msg.sender} ${msg.isError ? 'error' : ''}`}
            >
              <div className="message-content">
                {msg.sender === 'bot' ? (
                  <div className="bot-message">
                    <div className="answer-content">
                      {formatAnswer(msg.text)}
                    </div>
                    {msg.showMetadata && msg.metadata && formatMetadata(msg.metadata)}
                  </div>
                ) : (
                  msg.text
                )}
              </div>
            </motion.div>
          ))}
        </AnimatePresence>
        {loading && (
          <div className="loading-message">
            <CircularProgress size={20} />
            <span>AI is thinking...</span>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <div className="debug-controls">
        <button 
          className="debug-button"
          onClick={() => setShowAgentOutputs(!showAgentOutputs)}
        >
          {showAgentOutputs ? 'Hide' : 'Show'} Agent Outputs
        </button>
      </div>

      {showAgentOutputs && chatHistory.length > 0 && (
        <div className="agent-outputs">
          <div className="agent-output-section">
            <h4>User Agent Analysis</h4>
            <pre>
              {JSON.stringify(
                chatHistory[chatHistory.length - 1]?.agentOutputs?.user_analysis || {},
                null,
                2
              )}
            </pre>
          </div>
          <div className="agent-output-section">
            <h4>RAG Agent Prompt</h4>
            <pre>
              {chatHistory[chatHistory.length - 1]?.agentOutputs?.rag_prompt || 'No prompt available'}
            </pre>
          </div>
        </div>
      )}

      <div className="chat-input">
        <textarea
          value={input}
          onChange={handleInputChange}
          onKeyPress={handleKeyPress}
          placeholder="Ask a question..."
          rows={1}
        />
        <IconButton 
          onClick={handleSend}
          color="primary"
          className="send-button"
          disabled={loading || !input.trim()}
        >
          <SendIcon />
        </IconButton>
      </div>
    </motion.div>
  );
}

export default Chatbot; 