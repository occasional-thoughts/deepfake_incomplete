// src/useHashScroll.js

import { useEffect } from 'react';
import { useLocation } from 'react-router-dom';

const useHashScroll = (isLoaded) => {
  const location = useLocation();

  useEffect(() => {
    // We only want this to run when the content has finished loading.
    if (!isLoaded) {
      return;
    }

    const hash = location.hash;

    // Check if there is a hash and if we are on the homepage.
    if (hash && location.pathname === '/') {
      const id = hash.replace('#', '');
      const element = document.getElementById(id);

      if (element) {
        // Wait a fraction of a second to ensure all rendering is complete,
        // especially after the loader and with Framer Motion animations.
        setTimeout(() => {
          element.scrollIntoView({
            behavior: 'smooth',
            block: 'start', // Aligns the top of the element to the top of the viewport
          });
        }, 100);
      }
    }
  }, [isLoaded, location]); // Rerun this effect when isLoaded or location changes.
};

export default useHashScroll;