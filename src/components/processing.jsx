import React, { useEffect, useState, useRef } from 'react';
import { motion } from 'framer-motion';

// --- StrictMode duplicate-request guard (module-level, persists across remounts) ---
const activeKeys = new Set();
let globalAbortController = null;

const Processing = ({ onComplete, fileData, analysisType }) => {
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState(null);
  const [currentPhase, setCurrentPhase] = useState('Initializing...');
  const intervalRef = useRef(null);

  const clearExistingInterval = () => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  };

  const abortRequest = () => {
    if (globalAbortController) {
      globalAbortController.abort();
      globalAbortController = null;
    }
  };

  // Build a stable key for the current job
  const buildKey = () => {
    if (!fileData) return null;
    const lm = typeof fileData.lastModified === 'number' ? fileData.lastModified : 0;
    return `${fileData.name}|${fileData.size}|${lm}|${analysisType}`;
  };

  useEffect(() => {
    const key = buildKey();
    if (!key) return;

    if (activeKeys.has(key)) {
      return; // Prevent double run in StrictMode
    }
    activeKeys.add(key);

    performMediaAnalysis(key).finally(() => {
      activeKeys.delete(key);
    });

    return () => {
      clearExistingInterval();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fileData, analysisType]);

  const performMediaAnalysis = async (key) => {
    try {
      setError(null);
      clearExistingInterval();
      globalAbortController = new AbortController();

      // --- Phase 1: Validation ---
      setCurrentPhase('Validating file...');
      const maxSizeAudio = 50 * 1024 * 1024;
      const maxSizeVideo = 500 * 1024 * 1024;
      const maxSize = analysisType === 'video' ? maxSizeVideo : maxSizeAudio;

      if (fileData.size > maxSize) {
        throw new Error(`File too large. Maximum size: ${maxSize / (1024 * 1024)}MB`);
      }

      setProgress(15);
      await new Promise((r) => setTimeout(r, 800));

      // --- Phase 2: Upload ---
      setCurrentPhase('Uploading file...');
      setProgress(35);
      await new Promise((r) => setTimeout(r, 900));

      const formData = new FormData();
      formData.append('file', fileData);

      // --- Phase 3: Analysis ---
      const endpoint = analysisType === 'video' ? '/api/predict-video' : '/api/predict';
      const timeoutDuration = analysisType === 'video' ? 240000 : 120000; // 4min / 2min

      setCurrentPhase(
        analysisType === 'video' ? 'Analyzing video frames...' : 'Analyzing audio patterns...'
      );

      // creep progress until ~75% while waiting
      intervalRef.current = setInterval(() => {
        setProgress((prev) => {
          if (prev >= 75) {
            clearExistingInterval();
            return prev;
          }
          return prev + (analysisType === 'video' ? 0.3 : 0.7);
        });
      }, analysisType === 'video' ? 2000 : 800);

      const timeoutId = setTimeout(() => {
        abortRequest();
      }, timeoutDuration);

      const response = await fetch(`http://localhost:5001${endpoint}`, {
        method: 'POST',
        body: formData,
        signal: globalAbortController.signal,
      }).finally(() => {
        clearTimeout(timeoutId);
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const result = await response.json();
      clearExistingInterval();

      // --- Phase 4: Finalizing ---
      setCurrentPhase('Finalizing analysis...');
      setProgress(100); // ✅ immediately jump to 100 when result is ready

      setTimeout(() => {
        if (onComplete) onComplete(result);
      }, 500);
    } catch (err) {
      clearExistingInterval();
      let errorMessage;
      if (err.name === 'AbortError') {
        errorMessage = 'Request was cancelled';
      } else if (err.message && err.message.toLowerCase().includes('timeout')) {
        errorMessage =
          analysisType === 'video'
            ? 'Analysis timed out. Large videos may take longer to process.'
            : 'Analysis timed out. Please try again.';
      } else {
        errorMessage = err.message || 'Unknown error occurred';
      }

      setError(errorMessage);
      setCurrentPhase('Analysis failed');

      setTimeout(() => {
        if (onComplete) {
          onComplete({
            prediction: 'ERROR',
            confidence: 0,
            probabilities: { real: 0, fake: 0 },
            error: errorMessage,
          });
        }
      }, 800);
    } finally {
      globalAbortController = null;
    }
  };

  const handleCancel = () => {
    clearExistingInterval();
    abortRequest();
    if (onComplete) {
      onComplete({
        prediction: 'CANCELLED',
        confidence: 0,
        probabilities: { real: 0, fake: 0 },
        error: 'Request was cancelled by user',
      });
    }
  };

  return (
    <div className="w-screen min-h-screen pt-10 overflow-y-auto bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-white flex flex-col items-center justify-center font-sans relative">
      {/* Background */}
      <div className="fixed inset-0 overflow-hidden -z-10">
        <div className="absolute top-20 left-20 w-32 h-32 bg-cyan-400/10 rounded-full blur-xl animate-pulse"></div>
        <div className="absolute bottom-32 right-32 w-48 h-48 bg-blue-400/10 rounded-full blur-xl animate-pulse delay-1000"></div>
        <div className="absolute inset-0 bg-[linear-gradient(rgba(59,130,246,0.03)_1px,transparent_1px),linear-gradient(90deg,rgba(59,130,246,0.03)_1px,transparent_1px)] bg-[size:4rem_4rem]"></div>
      </div>

      {/* Content */}
      <div className="relative z-10 flex flex-col items-center justify-center w-full max-w-2xl px-4">
        {/* Loader Animation (replaced image with dual opposite circles) */}
          <div className="mb-12 relative">
            <div className="relative w-32 h-32 sm:w-40 sm:h-40 flex items-center justify-center">
              {/* Outer circle (clockwise) */}
              <div className="absolute inset-0 rounded-full border-4 border-transparent border-t-cyan-400 animate-spin"></div>

              {/* Inner circle (counterclockwise) */}
              <div className="absolute inset-4 rounded-full border-4 border-transparent border-b-blue-400 animate-spin-reverse"></div>

              {/* Glowing center pulse */}
              <div className="w-12 h-12 sm:w-16 sm:h-16 rounded-full bg-gradient-to-br from-cyan-400/30 to-blue-400/30 blur-sm animate-pulse"></div>
            </div>
          </div>


        <div className="text-center mb-8">
          <h1 className="text-3xl md:text-4xl font-bold mb-4 bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
            Analyzing Content
          </h1>

          {error ? (
            <div className="bg-red-500/20 border border-red-400/30 rounded-lg p-4 mb-6">
              <p className="text-red-400 text-sm font-semibold">⚠️ Analysis Failed</p>
              <p className="text-red-400 text-xs mt-2">{error}</p>
              <button
                onClick={handleCancel}
                className="mt-3 px-4 py-2 bg-red-600/20 hover:bg-red-600/30 border border-red-500/30 rounded-lg transition-colors text-red-300 text-sm"
              >
                Go Back
              </button>
            </div>
          ) : (
            <>
              <p className="text-base text-slate-300 mb-4">
                {analysisType === 'audio'
                  ? 'Our AI is analyzing your audio for deepfake patterns...'
                  : 'Our AI is analyzing your video for deepfake patterns. This may take several minutes...'}
              </p>
              <p className="text-sm text-cyan-400 mb-6">{currentPhase}</p>
            </>
          )}

          {/* Progress tracker */}
          <div className="flex justify-center items-center flex-wrap gap-4 mb-8 text-center">
            {['Upload', 'Analysis', 'Verification', 'Complete'].map((label, i) => (
              <div className="flex items-center gap-2" key={label}>
                <div
                  className={`w-3 h-3 rounded-full transition-colors duration-500 ${
                    progress >= [0, 20, 75, 100][i]
                      ? i === 3
                        ? 'bg-green-400'
                        : 'bg-cyan-400'
                      : 'bg-slate-600'
                  }`}
                ></div>
                <span className="text-xs sm:text-sm text-slate-400">{label}</span>
                {i < 3 && <div className="w-8 h-0.5 bg-slate-600 hidden sm:block"></div>}
              </div>
            ))}
          </div>
        </div>

        {/* Progress bar */}
        <div className="w-full max-w-md mb-8">
          <div className="relative mb-4">
            <div className="w-full h-3 bg-slate-700 rounded-full overflow-hidden shadow-inner">
              <motion.div
                className="h-full bg-gradient-to-r from-cyan-500 to-blue-500 rounded-full shadow-lg"
                style={{ width: `${Math.round(progress)}%` }} // ✅ force integers
                animate={{ width: `${Math.round(progress)}%` }} // ✅ smooth animate
                transition={{ duration: 0.5, ease: 'easeOut' }}
              />
            </div>
            <div className="absolute -top-8 left-0 right-0 text-center">
              <span className="text-2xl font-bold text-cyan-400">{Math.round(progress)}%</span>
            </div>
          </div>
        </div>

        {!error && (
          <div className="text-center">
            <button
              onClick={handleCancel}
              className="px-6 py-2 bg-slate-700/50 hover:bg-slate-600/50 border border-slate-600/30 rounded-lg transition-colors text-slate-300 text-sm"
            >
              Cancel Analysis
            </button>
            {analysisType === 'video' && (
              <p className="text-xs text-slate-500 mt-2">
                Video analysis typically takes 2–5 minutes depending on file size
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default Processing;
