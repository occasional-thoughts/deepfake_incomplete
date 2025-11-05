import React, { useState } from 'react';
import UploadModal from '../components/Upload';
import Processing from '../components/processing';
import AudioResult from '../components/AudioResult';
import VideoResult from '../components/VideoResult';

function DeepfakeUploadPage() {
  const [view, setView] = useState('upload');
  const [analysisType, setAnalysisType] = useState(null);
  const [fileData, setFileData] = useState(null);
  const [analysisResult, setAnalysisResult] = useState(null);

  const handleAnalysisStart = (fileType, file) => {
    console.log('Starting analysis:', { fileType, fileName: file.name, fileSize: file.size });
    setAnalysisType(fileType);
    setFileData(file);
    setView('processing');
  };

  const handleProcessingComplete = (result) => {
    console.log('Processing complete:', result);
    
    // Handle cancellation
    if (result.cancelled) {
      handleReset();
      return;
    }
    
    // Handle errors
    if (result.error || result.prediction === "ERROR") {
      console.error('Analysis error:', result.error);
      setAnalysisResult(result);
      
      // Show error in appropriate result component
      if (analysisType === 'audio') {
        setView('audioResult');
      } else if (analysisType === 'video') {
        setView('videoResult');
      }
      return;
    }
    
    // Handle successful analysis
    setAnalysisResult(result);
    
    if (analysisType === 'audio') {
      // Validate audio result structure
      if (!result.prediction && !result.probabilities) {
        console.warn('Audio result missing expected fields, using fallback');
        setAnalysisResult({
          ...result,
          prediction: result.prediction || 'real',
          probabilities: result.probabilities || { real: 0.5, fake: 0.5 }
        });
      }
      setView('audioResult');
    } else if (analysisType === 'video') {
      // Validate video result structure
      if (!result.video) {
        console.warn('Video result missing video field, creating fallback structure');
        setAnalysisResult({
          video: {
            overall_prediction: result.prediction || 'real',
            average_confidence: result.confidence || 0.5,
            detailed_results: {}
          },
          audio: result.audio || null
        });
      }
      setView('videoResult');
    }
  };

  const handleReset = () => {
    console.log('Resetting application state');
    setView('upload');
    setFileData(null);
    setAnalysisResult(null);
    setAnalysisType(null);
  };

  const renderCurrentView = () => {
    switch (view) {
      case 'processing':
        return (
          <Processing 
            onComplete={handleProcessingComplete}
            fileData={fileData}
            analysisType={analysisType}
          />
        );
      case 'audioResult':
        return (
          <AudioResult 
            onReset={handleReset}
            analysisResult={analysisResult}
          />
        );
      case 'videoResult':
        return (
          <VideoResult 
            onReset={handleReset} 
            analysisResult={analysisResult} 
          />
        );
      case 'upload':
      default:
        return (
          <UploadModal 
            onAnalysisComplete={handleAnalysisStart} 
          />
        );
    }
  };

  return (
    <div className="app-container">
      {renderCurrentView()}
    </div>
  );
}

export default DeepfakeUploadPage;