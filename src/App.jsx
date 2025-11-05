import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, useLocation } from 'react-router-dom';
import { AnimatePresence, motion } from 'framer-motion';
import useHashScroll from './components/useHashScroll';

// Core components only
import Loader from './components/loader';
import Hero from './components/hero';
import DeepfakeDetectionPlatform from './components/aboutCards';
import DeepfakeDetectionUpload from './pages/DeepFakeUpload';

const useActiveSection = (isLoaded) => {
  const [activeSection, setActiveSection] = useState('home');

  useEffect(() => {
    if (!isLoaded) return;

    const sections = ['home', 'about', 'products'];

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            setActiveSection(entry.target.id);
          }
        });
      },
      {
        rootMargin: '-50% 0px -50% 0px',
        threshold: 0
      }
    );

    sections.forEach(id => {
      const el = document.getElementById(id);
      if (el) observer.observe(el);
    });

    return () => observer.disconnect();
  }, [isLoaded]);

  useEffect(() => {
    if (isLoaded && activeSection) {
      window.history.replaceState(null, '', `/#${activeSection}`);
    }
  }, [activeSection, isLoaded]);

  return activeSection;
};

// Home section (no footer, no news)
const Home = ({ isLoaded, onFaceModelLoaded }) => {
  const activeSection = useActiveSection(isLoaded);

  return (
    <div className="z-10" style={{ visibility: isLoaded ? 'visible' : 'hidden' }}>
      <div id="home">
        <Hero
          Loaded={isLoaded}
          onFaceModelLoaded={onFaceModelLoaded}
          activeSection={activeSection}
          hideLogo
        />
      </div>

      <div id="products">
        <DeepfakeDetectionPlatform />
      </div>
    </div>
  );
};

// Wrapper for smooth page transitions
const PageWrapper = ({ children }) => (
  <motion.div
    initial={{ opacity: 0, y: 30 }}
    animate={{ opacity: 1, y: 0 }}
    exit={{ opacity: 0, y: -30 }}
    transition={{ duration: 0.6, ease: 'easeInOut' }}
  >
    {children}
  </motion.div>
);

const AppContent = () => {
  const location = useLocation();
  const [isLoaded, setIsLoaded] = useState(false);
  const [faceModelLoaded, setFaceModelLoaded] = useState(false);

  useHashScroll(isLoaded);

  const handleFaceModelLoaded = () => setFaceModelLoaded(true);
  const handleLoaderFinish = () => setIsLoaded(true);

  useEffect(() => {
    if (isLoaded && !location.hash) {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }
  }, [isLoaded, location]);

  return (
    <div className="relative bg-black overflow-hidden">
      {!isLoaded && (
        <Loader onFinish={handleLoaderFinish} faceModelLoaded={faceModelLoaded} />
      )}

      <AnimatePresence mode="wait">
        <Routes location={location} key={location.pathname}>
          <Route
            path="/deepfake-detection"
            element={
              <PageWrapper>
                <DeepfakeDetectionUpload />
              </PageWrapper>
            }
          />
          <Route
            path="*"
            element={
              <PageWrapper>
                <Home
                  isLoaded={isLoaded}
                  onFaceModelLoaded={handleFaceModelLoaded}
                />
              </PageWrapper>
            }
          />
        </Routes>
      </AnimatePresence>
    </div>
  );
};

const App = () => (
  <Router>
    <AppContent />
  </Router>
);

export default App;
