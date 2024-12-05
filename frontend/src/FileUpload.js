// src/FileUpload.js
import React, { useState } from 'react';
import { Box, Typography, CircularProgress } from '@mui/material';
import { useDropzone } from 'react-dropzone';

function FileUpload({ onUploadSuccess }) {
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);

  const onDrop = async (acceptedFiles) => {
    const file = acceptedFiles[0];
    if (!file) return;

    setUploading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:5000/upload', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Error uploading file');
      }

      // Call the success callback
      onUploadSuccess();
      
    } catch (err) {
      setError(err.message);
    } finally {
      setUploading(false);
    }
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf']
    },
    multiple: false
  });

  return (
    <Box
      {...getRootProps()}
      sx={{
        border: '2px dashed #6366f1',
        borderRadius: '10px',
        padding: '40px',
        textAlign: 'center',
        cursor: 'pointer',
        backgroundColor: isDragActive ? 'rgba(99, 102, 241, 0.1)' : 'transparent',
        transition: 'all 0.3s ease',
        '&:hover': {
          backgroundColor: 'rgba(99, 102, 241, 0.1)',
        }
      }}
    >
      <input {...getInputProps()} />
      {uploading ? (
        <Box display="flex" flexDirection="column" alignItems="center" gap={2}>
          <CircularProgress />
          <Typography>Uploading...</Typography>
        </Box>
      ) : (
        <Typography color={error ? 'error' : 'textSecondary'}>
          {error || (isDragActive
            ? 'Drop your PDF here'
            : 'Drag \'n\' drop some files here, or click to select files')}
        </Typography>
      )}
    </Box>
  );
}

export default FileUpload;