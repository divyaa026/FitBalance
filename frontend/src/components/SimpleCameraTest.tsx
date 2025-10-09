import React, { useRef, useEffect, useState } from 'react';

const SimpleCameraTest = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);

  const startCamera = async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({ video: true });
      setStream(mediaStream);
      
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
      }
    } catch (err) {
      console.error('Camera error:', err);
    }
  };

  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      setStream(null);
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  };

  return (
    <div className="p-4 border rounded-lg bg-white">
      <h3 className="text-lg font-semibold mb-4">Simple Camera Test</h3>
      
      <div className="mb-4">
        <video
          ref={videoRef}
          autoPlay
          muted
          playsInline
          className="w-full max-w-md h-64 bg-gray-800 border"
          style={{ objectFit: 'cover' }}
        />
      </div>
      
      <div className="space-x-2">
        <button
          onClick={startCamera}
          className="px-4 py-2 bg-blue-500 hover:bg-blue-600 text-white font-medium rounded transition-colors duration-200"
        >
          Start Camera
        </button>
        <button
          onClick={stopCamera}
          className="px-4 py-2 bg-red-500 hover:bg-red-600 text-white font-medium rounded transition-colors duration-200"
        >
          Stop Camera
        </button>
      </div>
      
      <div className="mt-2 text-sm text-gray-600">
        Stream status: {stream ? 'Active' : 'Inactive'}
      </div>
    </div>
  );
};

export default SimpleCameraTest;