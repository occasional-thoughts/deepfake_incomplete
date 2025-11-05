import React, { useState, useEffect } from 'react';
import { Shield, Mic, Video } from 'lucide-react';
import { useNavigate } from "react-router-dom";


const DeepfakeDetectionPlatform = () => {
  const [flipped, setFlipped] = useState(false);
  const [isLoaded, setIsLoaded] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    setIsLoaded(true);
  }, []);

  const service = {
    icon: Shield,
    bigTitle: "DEEPFAKE DETECTION",
    subtitle: "AI-powered detection for videos and audio to ensure content authenticity.",
    subServices: [
      { icon: Mic, name: "Audio Only" },
      { icon: Video, name: "Audio & Video" },
    ],
    buttonLabel: "Try Detection Tool",
    path: "/deepfake-detection",
    bgImage:
      "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 400 300'%3E%3Cdefs%3E%3ClinearGradient id='bg1' x1='0%25' y1='0%25' x2='100%25' y2='100%25'%3E%3Cstop offset='0%25' style='stop-color:%23064e77;stop-opacity:0.3'/%3E%3Cstop offset='100%25' style='stop-color:%23155e75;stop-opacity:0.1'/%3E%3C/linearGradient%3E%3C/defs%3E%3Crect width='400' height='300' fill='url(%23bg1)'/%3E%3Cg opacity='0.4'%3E%3Ccircle cx='80' cy='80' r='3' fill='%2306b6d4'/%3E%3Ccircle cx='320' cy='60' r='2' fill='%2306b6d4'/%3E%3Ccircle cx='150' cy='180' r='2.5' fill='%2306b6d4'/%3E%3Ccircle cx='280' cy='220' r='2' fill='%2306b6d4'/%3E%3Cpath d='M50 150 Q200 100 350 200' stroke='%2306b6d4' stroke-width='1' fill='none' opacity='0.3'/%3E%3Cpath d='M100 250 Q250 180 380 240' stroke='%2306b6d4' stroke-width='1' fill='none' opacity='0.2'/%3E%3C/g%3E%3Ctext x='200' y='160' text-anchor='middle' fill='%2306b6d4' font-family='Arial' font-size='24' font-weight='bold' opacity='0.1'%3EDETECT%3C/text%3E%3C/svg%3E",
  };

  const Icon = service.icon;

  const handleNavigate = (e) => {
    e.stopPropagation(); 
    navigate(service.path);
  };

  return (
    <div
      className="relative flex flex-col items-center justify-center min-h-screen bg-black text-white overflow-hidden px-4 py-8 sm:py-12"
      id="products"
    >
      {/* Header */}
      <div
        className={`text-center mb-8 sm:mb-10 md:mb-12 transition-all duration-1000 max-w-4xl ${
          isLoaded ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-6'
        }`}
      >
        <h1 className="text-3xl sm:text-4xl md:text-5xl lg:text-6xl font-extrabold text-cyan-400 mb-3 sm:mb-4 px-4">
          Deepfake Detection
        </h1>
        <p className="text-sm sm:text-base md:text-lg lg:text-xl text-gray-300 max-w-2xl mx-auto leading-relaxed px-4">
          Detect manipulated audio or video content with precision using our AI-powered verification engine.
        </p>
      </div>

      {/* Single Card */}
      <div
        className={`group relative w-full max-w-[340px] sm:max-w-[480px] md:max-w-[580px] lg:max-w-[700px] aspect-[4/3] sm:aspect-[3/2] [perspective:1200px] cursor-pointer transition-all duration-1000 ${
          isLoaded ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'
        }`}
        onClick={() => setFlipped(!flipped)}
      >
        <div
          className={`absolute inset-0 transition-transform duration-1000 [transform-style:preserve-3d] ${
            flipped ? '[transform:rotateY(180deg)]' : ''
          }`}
        >
          {/* Front Side */}
          <div
            className="absolute inset-0 rounded-2xl sm:rounded-3xl bg-slate-900/70 backdrop-blur-xl border border-slate-700/50 hover:border-cyan-500/40 transition-all duration-700 hover:scale-[1.02] sm:hover:scale-[1.03] flex flex-col items-center justify-center overflow-hidden [backface-visibility:hidden]"
            style={{
              backgroundImage: `url("${service.bgImage}")`,
              backgroundSize: 'cover',
              backgroundPosition: 'center',
            }}
          >
            <div className="absolute inset-0 bg-gradient-to-br from-slate-900/70 to-slate-800/60" />
            <div className="absolute inset-0 bg-gradient-to-r from-cyan-500/10 to-blue-500/10 opacity-0 group-hover:opacity-100 transition-opacity duration-700" />

            <div className="relative z-10 text-center px-4 sm:px-6 md:px-8">
              <div className="w-16 h-16 sm:w-20 sm:h-20 md:w-24 md:h-24 lg:w-28 lg:h-28 rounded-full bg-gradient-to-br from-cyan-500/30 to-cyan-600/20 flex items-center justify-center mb-4 sm:mb-5 md:mb-6 mx-auto border border-cyan-400/20">
                <Icon className="w-8 h-8 sm:w-10 sm:h-10 md:w-12 md:h-12 lg:w-14 lg:h-14 text-cyan-400" />
              </div>
              <h2 className="text-xl sm:text-2xl md:text-3xl lg:text-4xl font-black text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 via-blue-400 to-indigo-400 mb-3 sm:mb-4 leading-tight">
                {service.bigTitle}
              </h2>
              <p className="text-xs sm:text-sm md:text-base lg:text-lg text-gray-300 mb-4 sm:mb-6 md:mb-8 max-w-xs sm:max-w-sm md:max-w-md mx-auto leading-relaxed">
                {service.subtitle}
              </p>
              <div className="bg-slate-800/50 backdrop-blur-sm rounded-xl sm:rounded-2xl px-4 sm:px-5 md:px-6 py-2.5 sm:py-3 md:py-4 border border-cyan-500/20 inline-block">
                <p className="text-cyan-300 text-xs sm:text-sm font-medium uppercase tracking-widest">
                  Click for Details
                </p>
              </div>
            </div>
          </div>

          {/* Back Side */}
          <div className="absolute inset-0 rounded-2xl sm:rounded-3xl bg-slate-900/80 backdrop-blur-xl border border-cyan-400/40 shadow-2xl p-4 sm:p-5 md:p-6 lg:p-8 flex flex-col justify-between [transform:rotateY(180deg)] [backface-visibility:hidden]">
            <div className="flex-1 flex flex-col min-h-0">
              <div className="flex items-center gap-2 sm:gap-3 mb-3 sm:mb-4 md:mb-5">
                <div className="w-9 h-9 sm:w-11 sm:h-11 md:w-12 md:h-12 rounded-xl bg-cyan-500/20 flex items-center justify-center border border-cyan-400/20 flex-shrink-0">
                  <Icon className="w-5 h-5 sm:w-6 sm:h-6 text-cyan-400" />
                </div>
                <h3 className="text-base sm:text-lg md:text-xl lg:text-2xl font-bold text-white leading-tight">
                  {service.bigTitle}
                </h3>
              </div>

              <p className="text-gray-300 text-xs sm:text-sm md:text-base mb-3 sm:mb-4 md:mb-5 leading-snug sm:leading-relaxed">
                Robust AI models for detecting synthetic manipulations in media.
              </p>

              <div className="space-y-2 sm:space-y-2.5 md:space-y-3 mb-3 sm:mb-4 md:mb-5 flex-shrink-0">
                <h4 className="text-white font-semibold text-xs sm:text-sm md:text-base mb-2 sm:mb-2.5">
                  Detection Modes:
                </h4>
                {service.subServices.map((sub, idx) => {
                  const SubIcon = sub.icon;
                  return (
                    <div
                      key={idx}
                      className="flex items-center gap-2 sm:gap-3 p-2 sm:p-2.5 md:p-3 bg-slate-800/60 rounded-lg border border-slate-700"
                    >
                      <SubIcon className="w-4 h-4 sm:w-5 sm:h-5 text-cyan-400 flex-shrink-0" />
                      <span className="text-xs sm:text-sm md:text-base font-medium text-gray-200">
                        {sub.name}
                      </span>
                    </div>
                  );
                })}
              </div>

              <div className="mt-auto pt-2 sm:pt-3 flex-shrink-0">
                <button
                  className="w-full py-2.5 sm:py-3 md:py-3.5 px-4 sm:px-5 rounded-lg sm:rounded-xl bg-cyan-600/30 border border-cyan-400/40 text-white font-semibold text-xs sm:text-sm md:text-base lg:text-lg hover:bg-cyan-500/40 hover:border-cyan-300/60 transition-all duration-300 group/btn"
                  onClick={handleNavigate}
                >
                  <span className="flex items-center justify-center gap-2">
                    {service.buttonLabel}
                    <span className="inline-block group-hover/btn:translate-x-1 transition-transform duration-300">
                      →
                    </span>
                  </span>
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DeepfakeDetectionPlatform;