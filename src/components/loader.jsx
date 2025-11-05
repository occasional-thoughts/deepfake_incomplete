import React, { useEffect, useRef } from 'react';
import { gsap } from 'gsap';

/**
 * A component that displays a loading screen with a progress animation.
 * This version uses a single, pausable GSAP timeline for a more robust and smooth animation,
 * ensuring it always completes to 100% before finishing.
 */
const Loader = ({ onFinish, faceModelLoaded }) => {
  const progressRef = useRef(null);
  const containerRef = useRef(null);
  // Use a ref for the timeline to persist it across re-renders without causing side effects.
  const timelineRef = useRef(null);

  useEffect(() => {
    // Create the master timeline ONLY ONCE.
    if (!timelineRef.current) {
      const progress = { value: 0 };

      timelineRef.current = gsap.timeline({
        paused: true, // Start the timeline in a paused state.
        onComplete: () => {
          // The final onComplete for the entire sequence.
          // This triggers the fade-out animation.
          gsap.to(containerRef.current, {
            scale: 1.05,
            opacity: 0,
            duration: 0.8,
            ease: 'power2.inOut',
            onComplete: onFinish, // Call the final onFinish callback from App.jsx
          });
        }
      });

      // Part 1: Animate from 0% to 90%
      timelineRef.current.to(progress, {
        value: 90,
        duration: 2.5, // A consistent duration for the initial load
        ease: 'power1.out',
        onUpdate: () => {
          if (progressRef.current) {
            progressRef.current.textContent = `${Math.floor(progress.value)}%`;
          }
        },
      });

      // Part 2: Animate from 90% to 100%
      timelineRef.current.to(progress, {
        value: 100,
        duration: 0.5, // A quick completion once the model is loaded
        ease: 'power1.in',
        onUpdate: () => {
          if (progressRef.current) {
            progressRef.current.textContent = `${Math.floor(progress.value)}%`;
          }
        },
      });
    }

    // Start the animation as soon as the component mounts.
    // It will automatically pause after reaching 90% due to the timeline structure.
    timelineRef.current.play();

  }, [onFinish]); // Dependency array is now clean.


  // This separate effect is solely responsible for reacting to the face model loading.
  useEffect(() => {
    if (faceModelLoaded && timelineRef.current) {
      // If the model loads, resume the timeline. It will continue from where it paused (at 90%).
      timelineRef.current.resume();
    }
  }, [faceModelLoaded]);


  // Failsafe: If the model takes too long, gracefully complete the animation.
  useEffect(() => {
    const failsafeTimer = setTimeout(() => {
      if (timelineRef.current && timelineRef.current.paused()) {
        // If the timeline is still paused after 5 seconds, force it to resume.
        timelineRef.current.resume();
      }
    }, 5000); // 5-second timeout

    return () => clearTimeout(failsafeTimer);
  }, []);


  return (
    <div
      ref={containerRef}
      className="fixed scale-90 inset-0 bg-black text-cyan-400 flex items-center justify-center z-[9999] pointer-events-none"
      style={{ transformOrigin: 'center center' }}
    >
      <div className="text-center">
        <div className="absolute -translate-y-8 h-[70vh] inset-0 flex pointer-events-none flex-col items-center justify-center text-center z-10 bg-transparent bg-opacity-50">
          <h1 className="text-[10vw] text-white leading-[9vw] md:text-[5vw] lg:leading-[5vw] font-bold">
            Welcome to<br />
            <span className="bg-gradient-to-r from-[#6EE5F5] via-[#29A3B3] to-[#1397A9] bg-clip-text text-transparent">
              GenReal
            </span>.AI
          </h1>
          <p className="mt-4 text-lg text-gray-300">Discover the new age of security</p>
        </div>
        <div
          ref={progressRef}
          className="text-4xl mt-10 font-bold"
          style={{ userSelect: 'none' }}
        >
          0%
        </div>
        <div className="mt-2 text-white text-sm select-none">
          Loading GenReal.AI...
        </div>
      </div>
    </div>
  );
};

export default Loader;
