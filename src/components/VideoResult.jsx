import React from 'react';
import { motion } from 'framer-motion';
import { Home, AlertTriangle, Video, Brain, Users, Eye, Mic } from 'lucide-react';

// Helper function to truncate a number to a specified number of decimal places without rounding.
const truncate = (num, decimalPlaces) => {
  const factor = Math.pow(10, decimalPlaces);
  return Math.floor(num * factor) / factor;
};

const VideoResult = ({ onReset, analysisResult }) => {
  // --- Data Processing ---
  const videoData = analysisResult?.video;
  const audioData = analysisResult?.audio;
  const hasError = !!analysisResult?.error;

  // 1. Overall Assessment (for the main circle)
  const overallPrediction = videoData?.overall_prediction?.toLowerCase() || (hasError ? 'error' : 'real');
  const overallIsFake = overallPrediction === 'fake' || hasError;
  
  const overallConfidenceValue = videoData?.average_confidence || (hasError ? 0.87 : 0.13);
  const overallConfidence = truncate(overallConfidenceValue * 100, 2);

  const getConfidenceLevel = (confidence) => {
    if (confidence >= 85) return 'Very High';
    if (confidence >= 75) return 'High';
    if (confidence >= 60) return 'Medium';
    return 'Low';
  };

  const getResultColors = (isFake) => {
    if (isFake) {
      return {
        stroke: '#ef4444',
        badgeBg: 'bg-red-500/20',
        badgeBorder: 'border-red-400/30',
        text: 'text-red-400',
        progress: 'bg-red-500'
      };
    }
    return {
      stroke: '#22c55e',
      badgeBg: 'bg-green-500/20',
      badgeBorder: 'border-green-400/30',
      text: 'text-green-400',
      progress: 'bg-green-500'
    };
  };

  const overallColors = getResultColors(overallIsFake);

  // 2. Combine all dynamic analysis cards (Video per-person + Audio)
  let analysisCards = [];

  // 2a. Generate cards for each person detected in the video
  if (videoData?.detailed_results) {
    const personCards = Object.entries(videoData.detailed_results).map(([personId, personData], index) => {
      const totalBatches = personData.batches?.length || 0;
      const totalFakeProb = personData.batches?.reduce((sum, batch) => sum + (batch.fake_probability || 0), 0) || 0;
      const avgFakeProb = totalBatches > 0 ? totalFakeProb / totalBatches : 0;
      const displayConfidence = truncate(avgFakeProb * 100, 2);
      
      return {
        type: 'person',
        id: personId,
        name: `Person ${index + 1}`,
        confidence: displayConfidence,
        batchesCount: totalBatches,
        isFake: displayConfidence > 50,
        description: displayConfidence > 50 ? 'Deepfake indicators found in visual analysis.' : 'Visuals appear authentic.',
      };
    });
    analysisCards.push(...personCards);
  }

  // 2b. Generate card for audio analysis
  if (audioData) {
    const hasAudio = audioData.prediction !== 'NO AUDIO FOUND' && audioData.fake_probability != null;
    const isAudioFake = hasAudio && audioData.prediction.toLowerCase() === 'fake';
    const audioConfidence = hasAudio ? truncate(audioData.fake_probability * 100, 2) : 0;

    analysisCards.push({
      type: 'audio',
      id: 'audio-analysis',
      name: 'Audio Analysis',
      confidence: audioConfidence,
      isFake: isAudioFake,
      hasAudio: hasAudio,
      description: hasAudio
        ? (isAudioFake ? 'Audio manipulation patterns detected.' : 'Audio appears natural and unmodified.')
        : 'No audio track was found in the file.',
    });
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 to-blue-950 text-cyan-100 flex flex-col items-center justify-center p-6 relative overflow-hidden">
      {/* Background glowing circles */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-20 left-10 w-32 h-32 bg-cyan-500/10 rounded-full blur-xl"></div>
        <div className="absolute bottom-40 right-20 w-24 h-24 bg-cyan-400/15 rounded-full blur-lg"></div>
      </div>

      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.7, ease: "easeOut" }}
        className="w-full max-w-5xl text-center relative z-10"
      >
        {/* Header */}
        <motion.h1
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5, ease: "easeOut", delay: 0.2 }}
          className="text-4xl font-bold mb-4 text-cyan-300"
        >
          Analysis Result
        </motion.h1>

        {/* Overall Confidence Circle */}
        <motion.div
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          transition={{ type: "spring", stiffness: 200, delay: 0.4 }}
          className="mb-8"
        >
          <div className="w-32 h-32 mx-auto mb-4 relative">
            <svg className="w-32 h-32 transform -rotate-90" viewBox="0 0 100 100">
              <circle cx="50" cy="50" r="42" fill="transparent" stroke="currentColor" strokeWidth="4" className="text-slate-800" />
              <motion.circle
                cx="50" cy="50" r="42" fill="transparent" stroke={overallColors.stroke}
                strokeWidth="4" strokeLinecap="round" strokeDasharray={2 * Math.PI * 42}
                initial={{ strokeDashoffset: 2 * Math.PI * 42 }}
                animate={{ strokeDashoffset: 2 * Math.PI * 42 * (1 - overallConfidence / 100) }}
                transition={{ duration: 2.5, ease: "easeOut", delay: 1.2 }}
              />
            </svg>
            <div className="absolute inset-0 flex items-center justify-center flex-col">
              <motion.span
                initial={{ opacity: 0, scale: 0.5 }} animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.8, ease: "easeOut", delay: 1.5 }}
                className="text-2xl font-bold text-cyan-300"
              >
                {overallConfidence.toFixed(2)}%
              </motion.span>
            </div>
          </div>

          <div className={`inline-flex items-center gap-2 px-4 py-2 ${overallColors.badgeBg} ${overallColors.badgeBorder} border rounded-full mb-2`}>
            {overallIsFake ? <AlertTriangle className={`w-4 h-4 ${overallColors.text}`} /> : <Brain className={`w-4 h-4 ${overallColors.text}`} />}
            <span className={`${overallColors.text} font-medium`}>
              {hasError ? 'ERROR' : (overallIsFake ? 'DEEPFAKE DETECTED' : 'AUTHENTIC')}
            </span>
          </div>
          <p className="text-slate-400 text-sm">
            Overall Confidence: <span className={`${overallColors.text} font-semibold`}>{getConfidenceLevel(overallConfidence)}</span>
          </p>
        </motion.div>

        {/* Dynamic Analysis Cards */}
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
          {analysisCards.map((card, index) => {
            const cardColors = getResultColors(card.isFake);
            const showConfidence = !(card.type === 'audio' && !card.hasAudio);

            return (
              <motion.div
                key={card.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 * index + 0.5 }}
                className="bg-slate-900/50 backdrop-blur-sm p-6 rounded-xl border border-cyan-800/30 hover:border-cyan-600/50 transition-all duration-300 flex flex-col"
              >
                <div className={`text-2xl mb-3 ${showConfidence ? cardColors.text : 'text-slate-500'}`}>
                  {card.type === 'audio' ? <Mic className="w-6 h-6" /> : <Users className="w-6 h-6" />}
                </div>
                <h3 className="text-lg font-semibold mb-2 text-cyan-200">{card.name}</h3>
                <div className={`text-2xl font-bold ${showConfidence ? cardColors.text : 'text-slate-500'} mb-2`}>
                  {showConfidence ? `${card.confidence.toFixed(2)}%` : 'N/A'}
                </div>
                <p className="text-sm text-slate-400 mb-4 flex-grow">{card.description}</p>
                <div className="w-full bg-slate-800 rounded-full h-2 mb-4">
                  {showConfidence && (
                    <motion.div
                      className={`h-2 rounded-full ${cardColors.progress}`}
                      initial={{ width: 0 }}
                      animate={{ width: `${card.confidence}%` }}
                      transition={{ delay: 0.1 * index + 1, duration: 1 }}
                    />
                  )}
                </div>
                <div className="flex items-center justify-center gap-2 text-xs text-slate-300">
                  {card.type === 'person' ? <><Eye className="w-4 h-4" /><span>{card.batchesCount} batch(es)</span></> : <><Brain className="w-4 h-4" /><span>{card.hasAudio ? 'Analyzed' : 'No Audio'}</span></>}
                </div>
              </motion.div>
            );
          })}
        </div>
        
        {/* Buttons */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, ease: "easeOut", delay: 0.8 }}
          className="flex flex-col sm:flex-row gap-4 justify-center"
        >
          <button onClick={() => window.location.href = '/'} className="flex items-center gap-3 px-6 py-3 bg-slate-700/50 hover:bg-slate-600/50 border border-slate-600/30 rounded-lg transition-colors text-cyan-200">
            <Home className="w-4 h-4" /><span>Back to Home</span>
          </button>
          <button onClick={onReset} className="flex items-center gap-3 px-6 py-3 bg-cyan-600/20 hover:bg-cyan-600/30 border border-cyan-500/30 rounded-lg transition-colors text-cyan-300">
            <Video className="w-4 h-4" /><span>Analyze Another</span>
          </button>
        </motion.div>
      </motion.div>
    </div>
  );
};

export default VideoResult;