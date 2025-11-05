import React, { Suspense, useRef, useEffect, useState } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { useGLTF } from '@react-three/drei';
import * as THREE from 'three';

// A simple debounce function helper
const debounce = (func, delay) => {
  let timeoutId;
  return function() {
    const context = this;
    const args = arguments;
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => func.apply(context, args), delay);
  }
};

const Face = ({ isParentInView, paused, onModelLoaded }) => {
  const { scene } = useGLTF('/RoboFace/scene.gltf');
  const groupRef = useRef();
  const [scale, setScale] = useState({ x: 1.5, y: 1.2, z: 1.2 });
  
  const hasCalledLoaded = useRef(false);

  // Call the loaded callback once
  useEffect(() => {
    if (isParentInView && !hasCalledLoaded.current) {
      onModelLoaded?.();
      hasCalledLoaded.current = true;
    }
  }, [isParentInView, onModelLoaded]);

  // Responsive scaling
  useEffect(() => {
    const handleResize = () => {
      const width = window.innerWidth;
      let newScale, newY;
      if (width < 480) {
        newScale = { x: 0.95, y: 0.8, z: 1 }; newY = -0.85;
      } else if (width < 768) {
        newScale = { x: 1, y: 0.8, z: 1 }; newY = -0.9;
      } else if (width < 1024) {
        newScale = { x: 1.3, y: 1, z: 1 }; newY = -1.4;
      } else {
        newScale = { x: 1.45, y: 1.15, z: 1.1 }; newY = -1.6;
      }
      setScale(newScale);
      scene.position.y = newY;
    };

    const debouncedResize = debounce(handleResize, 250);
    handleResize(); // initial call
    window.addEventListener('resize', debouncedResize);
    return () => window.removeEventListener('resize', debouncedResize);
  }, [scene]);

  // Apply scale
  useEffect(() => {
    scene.scale.set(scale.x, scale.y, scale.z);
  }, [scale, scene]);

  // Wireframe overlay (optional, same as before)
  useEffect(() => {
    if (!isParentInView) return;

    const wireframeMaterial = new THREE.MeshBasicMaterial({
      color: 'grey',
      wireframe: true,
      transparent: true,
      opacity: 0.8,
      side: THREE.DoubleSide
    });

    const wireframeMeshes = [];
    scene.traverse((child) => {
      if (child.isMesh && child.geometry) {
        const wireframeMesh = new THREE.Mesh(child.geometry, wireframeMaterial);
        wireframeMesh.position.copy(child.position);
        wireframeMesh.rotation.copy(child.rotation);
        wireframeMesh.scale.copy(child.scale);
        if (child.parent) {
          child.parent.add(wireframeMesh);
          wireframeMeshes.push(wireframeMesh);
        }
      }
    });

    return () => {
      wireframeMeshes.forEach(mesh => mesh.parent?.remove(mesh));
      wireframeMaterial.dispose();
    };
  }, [scene, isParentInView]);

  // Animation frame (removed mouse tracking)
  useFrame(() => {
    if (paused || !groupRef.current) return;
    // Optionally, you could add some idle rotation here if you want
    // groupRef.current.rotation.y += 0.001;
  });

  return <group ref={groupRef}><primitive object={scene} /></group>;
};

const FaceModel = ({ paused, onModelLoaded }) => {
  return (
    <Canvas
      dpr={[1, 1.5]}
      className="absolute inset-0 pointer-events-none z-20"
      camera={{ position: [0, 0, 5], fov: 35 }}
      shadows
    >
      <ambientLight intensity={1} />
      <directionalLight position={[2, 2, 5]} intensity={1.2} />
      <Suspense fallback={null}>
        <Face 
          isParentInView={!paused} 
          paused={paused}
          onModelLoaded={onModelLoaded}
        />
      </Suspense>
    </Canvas>
  );
};

export default FaceModel;
