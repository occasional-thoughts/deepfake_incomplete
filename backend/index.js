// server.js (Express backend)
const express = require('express');
const cors = require('cors');
const multer = require('multer');
const axios = require('axios');
const FormData = require('form-data');
require('dotenv').config();

const app = express();

// Configure multer for file uploads
const upload = multer({
  limits: {
    fileSize: 100 * 1024 * 1024, // 100MB limit
  },
});

// Enable CORS for your frontend
app.use(cors({
  origin: ['http://localhost:3000', 'http://localhost:5173'],
  credentials: true
}));

app.use(express.json());

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ status: 'OK', timestamp: new Date().toISOString() });
});

// Helper function to determine file type
const getFileType = (mimetype, filename) => {
  if (mimetype) {
    if (mimetype.startsWith('audio/')) return 'audio';
    if (mimetype.startsWith('video/')) return 'video';
  }

  // Fallback to file extension
  const ext = filename.toLowerCase().split('.').pop();
  const audioExts = ['mp3', 'wav', 'flac', 'm4a', 'aac', 'ogg'];
  const videoExts = ['mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm'];

  if (audioExts.includes(ext)) return 'audio';
  if (videoExts.includes(ext)) return 'video';

  return 'unknown';
};

// Shared prediction handler (avoids internal re-routing)
async function handlePrediction(req, res, kind) {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No file uploaded' });
    }

    const fileType = getFileType(req.file.mimetype, req.file.originalname);

    if (kind === 'audio' && fileType !== 'audio') {
      return res.status(400).json({
        error: 'Invalid file type for audio endpoint. Please use audio files only.'
      });
    }
    if (kind === 'video' && fileType !== 'video') {
      return res.status(400).json({
        error: 'Invalid file type for video endpoint. Please use video files only.'
      });
    }
    if (kind === 'auto' && (fileType !== 'audio' && fileType !== 'video')) {
      return res.status(400).json({
        error: `Unsupported file type: ${fileType}. Please upload audio or video files only.`,
        supportedFormats: {
          audio: ['mp3', 'wav', 'flac', 'm4a', 'aac', 'ogg'],
          video: ['mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm']
        }
      });
    }

    console.log(`${(kind === 'auto' ? fileType : kind).toUpperCase()} file details:`, {
      originalname: req.file.originalname,
      mimetype: req.file.mimetype,
      size: req.file.size,
      type: fileType
    });

    if (!process.env.TOKEN_ID || !process.env.TOKEN_SECRET) {
      console.error('Missing API credentials');
      return res.status(500).json({ error: 'Server configuration error - missing API credentials' });
    }

    // Build form-data
    const formData = new FormData();
    formData.append('file', req.file.buffer, {
      filename: req.file.originalname,
      contentType: req.file.mimetype || (fileType === 'video' ? 'video/mp4' : 'audio/mpeg')
    });

    // Decide target based on resolved type
    const resolvedType = kind === 'auto' ? fileType : kind;
    const isVideo = resolvedType === 'video';
    const url = isVideo
      ? 'https://binshilin63--og-deepfake-detector-originaldeepfakeapi-predict.modal.run'
      : 'https://binshilin63--deepfake-detector-deepfakeaudioapi-predict.modal.run';

    console.log(`Sending request to ${isVideo ? 'Video' : 'Audio'} Modal API...`);

    const response = await axios.post(
      url,
      formData,
      {
        headers: {
          'Modal-Key': process.env.TOKEN_ID,
          'Modal-Secret': process.env.TOKEN_SECRET,
          ...formData.getHeaders()
        },
        timeout: isVideo ? 300000 : 180000,
        maxContentLength: 100 * 1024 * 1024,
        maxBodyLength: 100 * 1024 * 1024
      }
    );

    console.log(`${isVideo ? 'Video' : 'Audio'} Modal API response:`, response.status);
    console.log('🔍 API Result:', JSON.stringify(response.data, null, 2));

    console.log(`${isVideo ? 'Video' : 'Audio'} Modal API response:`, response.status);
    res.json(response.data);

  } catch (error) {
    console.error(`${kind} proxy error:`, error.message);
    handleApiError(error, res, kind);
  }
}

// Audio prediction endpoint
app.post('/api/predict', upload.single('file'), async (req, res) => {
  await handlePrediction(req, res, 'audio');
});

// Video prediction endpoint
app.post('/api/predict-video', upload.single('file'), async (req, res) => {
  await handlePrediction(req, res, 'video');
});

// Optional: Auto endpoint (safe – no internal re-routing or double multer)
app.post('/api/predict-auto', upload.single('file'), async (req, res) => {
  await handlePrediction(req, res, 'auto');
});

// Error handling helper
const handleApiError = (error, res, type) => {
  console.error(`${type} API error details:`, {
    message: error.message,
    status: error.response?.status,
    data: error.response?.data,
    config: {
      url: error.config?.url,
      method: error.config?.method,
      headers: error.config?.headers ? Object.keys(error.config.headers) : []
    }
  });

  if (error.code === 'ECONNABORTED') {
    return res.status(408).json({
      error: `Request timeout - ${type} analysis took too long`,
      details: 'The file may be too large or the service is experiencing high load'
    });
  }

  if (error.response) {
    return res.status(error.response.status).json({
      error: error.response.data || `${type} API Error: ${error.response.status}`,
      details: error.response.statusText
    });
  } else if (error.request) {
    return res.status(503).json({
      error: `Unable to reach ${type} analysis service`,
      details: 'The service may be temporarily unavailable'
    });
  } else {
    return res.status(500).json({
      error: 'Internal server error',
      details: error.message,
      type: type
    });
  }
};

const PORT = process.env.PORT || 5001;
app.listen(PORT, () => {
  console.log(`🚀 Proxy server running on port ${PORT}`);
  console.log(`📍 Health check: http://localhost:${PORT}/health`);
  console.log(`🔗 Audio API endpoint: http://localhost:${PORT}/api/predict`);
  console.log(`🔗 Video API endpoint: http://localhost:${PORT}/api/predict-video`);
  console.log(`🔗 Auto-detect endpoint: http://localhost:${PORT}/api/predict-auto`);
});
