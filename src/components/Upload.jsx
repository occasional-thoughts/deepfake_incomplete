import React, { useState, useEffect, useRef, useCallback } from 'react';
import { gsap } from 'gsap';
import { FiUploadCloud, FiMic, FiVideo, FiAlertTriangle } from 'react-icons/fi';

const UploadModal = ({ onAnalysisComplete }) => {
  const [activeTab, setActiveTab] = useState('audio'); // default tab
  const [dragging, setDragging] = useState(false);
  const [file, setFile] = useState(null);
  const [toastMessage, setToastMessage] = useState(null);
  const [linkInput, setLinkInput] = useState('');
  const [showFormatError, setShowFormatError] = useState(false);
  const [formatErrorMessage, setFormatErrorMessage] = useState('');

  const fileInputRef = useRef(null);
  const modalRef = useRef(null);

  const fileTypes = {
    audio: {
      accept: '.mp3,.wav',
      formats: ['mp3', 'wav'],
      description: 'Audio: MP3, WAV',
      maxSize: '50MB',
      type: 'audio'
    },
    video: {
      accept: '.mp4',
      formats: ['mp4'],
      description: 'Video: MP4',
      maxSize: '100MB',
      type: 'video'
    }
  };

  useEffect(() => {
    gsap.fromTo(
      modalRef.current,
      { x: 200, opacity: 0 },
      { x: 0, opacity: 1, duration: 1.8, ease: 'power3.out' }
    );
  }, []);

  useEffect(() => {
    const handlePaste = (e) => {
      const items = e.clipboardData.items;
      for (let item of items) {
        if (item.kind === 'file') {
          const blob = item.getAsFile();
          if (validateFileType(blob)) {
            setFile(blob);
            setToastMessage("📋 File pasted from clipboard!");
            setTimeout(() => setToastMessage(null), 3000);
          }
          break;
        }
      }
    };
    window.addEventListener("paste", handlePaste);
    return () => window.removeEventListener("paste", handlePaste);
  }, [activeTab]);

  const validateFileType = (file) => {
    const fileExtension = file.name.split('.').pop().toLowerCase();
    const allowedFormats = fileTypes[activeTab].formats;
    if (!allowedFormats.includes(fileExtension)) {
      showFormatErrorPopup(fileExtension, allowedFormats);
      return false;
    }
    return true;
  };

  const showFormatErrorPopup = (fileExt, allowedFormats) => {
    setFormatErrorMessage(`Invalid file format: .${fileExt}\nPlease upload ${allowedFormats.map(f => `.${f}`).join(', ')} files only.`);
    setShowFormatError(true);
  };
  
  const closeFormatError = () => setShowFormatError(false);

  const handleDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile && validateFileType(droppedFile)) {
        setFile(droppedFile);
        setToastMessage("📁 File Dropped!");
        setTimeout(() => setToastMessage(null), 3000);
    }
  };

  const handleFileSelect = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile && validateFileType(selectedFile)) {
      setFile(selectedFile);
    }
  };

  const handleTabChange = (tab) => {
    setActiveTab(tab);
    setFile(null);
    setLinkInput('');
  };

  const handleConfirm = async () => {
    if (!file && !linkInput.trim()) {
      setToastMessage("⚠️ Please select a file or enter a link");
      setTimeout(() => setToastMessage(null), 3000);
      return;
    }

    let fileToAnalyze = file;

    // If user provided a link, we need to fetch it
    if (!file && linkInput.trim()) {
      try {
        setToastMessage("📥 Downloading file from link...");
        const response = await fetch(linkInput);
        if (!response.ok) {
          throw new Error('Failed to download file');
        }
        const blob = await response.blob();
        
        // Create a File object from the blob
        const fileName = linkInput.split('/').pop() || `file.${fileTypes[activeTab].formats[0]}`;
        fileToAnalyze = new File([blob], fileName, { type: blob.type });
        
        setToastMessage("✅ File downloaded successfully!");
      } catch (error) {
        setToastMessage("❌ Failed to download file from link");
        setTimeout(() => setToastMessage(null), 3000);
        return;
      }
    }

    if (onAnalysisComplete) {
      const fileType = fileTypes[activeTab].type;
      onAnalysisComplete(fileType, fileToAnalyze);
    }
  };

  const getTabIcon = (tab) => {
    switch (tab) {
      case 'audio': return <FiMic className="w-5 h-5" />;
      case 'video': return <FiVideo className="w-5 h-5" />;
      default: return <FiUploadCloud className="w-5 h-5" />;
    }
  };

  return (
    <div className="w-screen h-screen flex items-center justify-center bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 relative font-exo text-white overflow-hidden">
      {/* Background */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute top-20 left-20 w-32 h-32 bg-cyan-400/10 rounded-full blur-xl animate-pulse"></div>
        <div className="absolute bottom-32 right-32 w-48 h-48 bg-blue-400/10 rounded-full blur-xl animate-pulse delay-1000"></div>
        <div className="absolute top-1/2 left-1/4 w-24 h-24 bg-teal-400/10 rounded-full blur-xl animate-pulse delay-2000"></div>
        <div className="absolute inset-0 bg-[linear-gradient(rgba(59,130,246,0.03)_1px,transparent_1px),linear-gradient(90deg,rgba(59,130,246,0.03)_1px,transparent_1px)] bg-[size:4rem_4rem]"></div>
      </div>

      {/* Modal */}
      <div
        ref={modalRef}
        className="relative bg-slate-800/90 backdrop-blur-sm border border-cyan-400/30 rounded-3xl px-6 sm:px-10 md:px-12 py-8 w-[95vw] max-w-[900px] shadow-2xl z-10"
      >
        <button
          onClick={() => window.location.replace('/')}
          className="absolute top-3 right-4 text-white text-2xl font-bold hover:text-cyan-400 transition"
        >
          &times;
        </button>

        <h2 className="text-3xl font-bold mb-8 text-center bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
          Upload your content
        </h2>

        {/* Tab Navigation */}
        <div className="flex justify-center mb-8">
          <div className="bg-slate-700/50 rounded-2xl p-2 flex space-x-2">
            {Object.keys(fileTypes).map((tab) => (
              <button
                key={tab}
                onClick={() => handleTabChange(tab)}
                className={`flex items-center space-x-2 px-6 py-3 rounded-xl font-semibold transition-all duration-300 ${
                  activeTab === tab
                    ? 'bg-gradient-to-r from-cyan-500 to-blue-600 text-white shadow-lg'
                    : 'text-slate-300 hover:text-white hover:bg-slate-600/50'
                }`}
              >
                {getTabIcon(tab)}
                <span className="capitalize">{tab}</span>
              </button>
            ))}
          </div>
        </div>

        {/* Dropzone */}
        <div
          onDrop={handleDrop}
          onDragOver={(e) => { e.preventDefault(); }}
          className="border-2 border-dashed rounded-xl p-8 mb-6 text-center transition-all duration-300 font-inter border-cyan-400/50 bg-slate-700/30"
        >
          <label htmlFor="file-upload" className="cursor-pointer block">
            <input
              ref={fileInputRef}
              id="file-upload"
              type="file"
              accept={fileTypes[activeTab].accept}
              onChange={handleFileSelect}
              className="hidden"
            />
            <div className="mb-4">
              <FiUploadCloud className="text-cyan-400 text-6xl mx-auto animate-pulse" />
            </div>
            <p className="font-semibold text-lg mb-2">
              {file ? (
                <span className="text-green-400">📁 {file.name}</span>
              ) : (
                `Drag and drop your ${activeTab} here`
              )}
            </p>
          </label>
        </div>

        {/* Separator & Link Input */}
        <div className="text-center text-slate-400 mb-6 relative">
          <div className="absolute inset-0 flex items-center">
            <div className="w-full border-t border-slate-600"></div>
          </div>
          <div className="relative bg-slate-800 px-4 text-sm">or paste a link</div>
        </div>

        <div className="mb-8">
          <input
            type="text"
            className="w-full px-4 py-3 rounded-xl bg-slate-700/50 text-white placeholder-slate-400 border border-cyan-400/30 focus:outline-none focus:ring-2 focus:ring-cyan-500"
            placeholder={`https://example.com/your-${activeTab}.${fileTypes[activeTab].formats[0]}`}
            value={linkInput}
            onChange={(e) => setLinkInput(e.target.value)}
          />
        </div>

        {/* Action Buttons */}
        <div className="flex justify-end gap-4">
          <button
            className="bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-600 hover:to-blue-700 text-white px-8 py-3 rounded-xl font-bold transition-all duration-300 hover:scale-105 shadow-lg disabled:opacity-50 disabled:cursor-not-allowed"
            onClick={handleConfirm}
            disabled={!file && !linkInput.trim()}
          >
            Analyze {activeTab}
          </button>
        </div>

        {/* Toast Message */}
        {toastMessage && (
          <div className="fixed bottom-4 right-4 bg-cyan-600 text-white px-4 py-2 rounded-lg shadow-lg z-50">
            {toastMessage}
          </div>
        )}

        {/* Format Error Modal */}
        {showFormatError && (
          <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
            <div className="bg-slate-800 border border-red-400/30 rounded-lg p-6 max-w-sm">
              <div className="flex items-center gap-2 mb-3">
                <FiAlertTriangle className="text-red-400" />
                <h3 className="text-red-400 font-semibold">Invalid File Format</h3>
              </div>
              <p className="text-slate-300 mb-4 whitespace-pre-line">{formatErrorMessage}</p>
              <button
                onClick={closeFormatError}
                className="w-full bg-red-600 hover:bg-red-700 text-white py-2 rounded-lg transition"
              >
                OK
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default UploadModal;