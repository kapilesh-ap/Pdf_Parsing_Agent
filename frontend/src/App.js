// src/App.js
import React, { useState } from 'react';
import { CircularProgress, Typography, Box } from '@mui/material';
import FileUpload from './FileUpload';
import Chatbot from './Chatbot';
import './App.css';

function App() {
  const [isFileUploaded, setIsFileUploaded] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  const handleFileUploadSuccess = () => {
    setIsLoading(true);
    // Simulate loading for smoother transition
    setTimeout(() => {
      setIsFileUploaded(true);
      setIsLoading(false);
    }, 2000);
  };

  return (
    <div className="app-container">
      {!isFileUploaded ? (
        <div className="landing-container">
          <Typography variant="h2" className="main-title">
            PDF Chat Assistant
          </Typography>
          <Typography variant="h5" className="subtitle">
            Upload your PDF to start the conversation
          </Typography>
          <Box className="upload-section">
            <FileUpload onUploadSuccess={handleFileUploadSuccess} />
          </Box>
          {isLoading && (
            <Box className="loading-overlay">
              <CircularProgress size={60} />
              <Typography variant="h6" className="loading-text">
                Preparing your chat assistant...
              </Typography>
            </Box>
          )}
          <Typography variant="body2" className="attribution">
            Created by Kapileshvar
          </Typography>
        </div>
      ) : (
        <Chatbot />
      )}
    </div>
  );
}

export default App;