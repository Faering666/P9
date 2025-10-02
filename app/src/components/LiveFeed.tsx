'use client';

import { useState, useEffect, useRef } from 'react';

interface ImageData {
  filename: string;
  timestamp: string;
}

export default function LiveFeed() {
  const [images, setImages] = useState<ImageData[]>([]);
  const [selectedImage, setSelectedImage] = useState<ImageData | null>(null);
  const [loading, setLoading] = useState(true);
  const [webcamError, setWebcamError] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    // Fetch the images list from the JSON file
    fetch('/satellite-images/images.json')
      .then(res => res.json())
      .then((data: ImageData[]) => {
        setImages(data);
        setLoading(false);
      })
      .catch(error => {
        console.error('Error loading images:', error);
        setLoading(false);
      });
  }, []);

  useEffect(() => {
    // Access webcam
    async function startWebcam() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 1920 },
            height: { ideal: 1080 }
          }
        });

        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (error) {
        console.error('Error accessing webcam:', error);
        setWebcamError('Unable to access webcam. Please grant camera permissions.');
      }
    }

    startWebcam();

    // Cleanup function to stop webcam when component unmounts
    return () => {
      if (videoRef.current && videoRef.current.srcObject) {
        const stream = videoRef.current.srcObject as MediaStream;
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <p className="text-lg text-gray-600 dark:text-gray-400">Loading...</p>
      </div>
    );
  }

  return (
    <div className="relative w-full h-[calc(100vh-120px)]">
      {/* Selected Image - Top Right */}
      {selectedImage && (
        <div className="absolute top-4 right-4 z-10 w-80 bg-white dark:bg-gray-800 rounded-lg shadow-2xl overflow-hidden border-2 border-blue-500">
          <div className="relative">
            <img
              src={`/satellite-images/${selectedImage.filename}`}
              alt={`Selected: ${selectedImage.timestamp}`}
              className="w-full h-48 object-cover"
            />
            <div className="absolute bottom-0 left-0 right-0 bg-black/70 text-white px-3 py-2">
              <p className="text-sm font-mono">{selectedImage.timestamp}</p>
            </div>
            <button
              onClick={() => setSelectedImage(null)}
              className="absolute top-2 right-2 bg-red-500 hover:bg-red-600 text-white rounded-full w-8 h-8 flex items-center justify-center font-bold"
            >
              ×
            </button>
          </div>
        </div>
      )}

      {/* Live Feed - Main View */}
      <div className="w-full h-full bg-gray-900 rounded-lg overflow-hidden flex items-center justify-center">
        <div className="relative w-full h-full">
          {/* Webcam video feed */}
          {webcamError ? (
            <div className="absolute inset-0 bg-gradient-to-br from-gray-800 to-gray-900 flex items-center justify-center">
              <div className="text-center max-w-md px-4">
                <p className="text-red-500 text-xl font-mono mb-2">Camera Access Error</p>
                <p className="text-gray-400 text-sm">{webcamError}</p>
              </div>
            </div>
          ) : (
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="w-full h-full object-cover"
            />
          )}

          {/* Live indicator */}
          <div className="absolute top-4 left-4 bg-red-600 text-white px-3 py-1 rounded-full flex items-center gap-2">
            <div className="w-2 h-2 bg-white rounded-full animate-pulse"></div>
            <span className="text-sm font-bold">LIVE</span>
          </div>
        </div>
      </div>

      {/* Image Selection Grid - Bottom Overlay */}
      <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent p-4">
        <div className="flex gap-3 overflow-x-auto pb-2">
          {images.map((image, index) => (
            <button
              key={index}
              onClick={() => setSelectedImage(image)}
              className={`flex-shrink-0 flex flex-col items-center gap-1 p-2 rounded-lg transition-all ${
                selectedImage?.filename === image.filename
                  ? 'bg-blue-500 shadow-lg'
                  : 'bg-gray-800/80 hover:bg-gray-700/80'
              }`}
            >
              <div className="relative w-28 h-20 bg-gray-700 rounded overflow-hidden">
                <img
                  src={`/satellite-images/${image.filename}`}
                  alt={`Thumbnail ${index + 1}`}
                  className="w-full h-full object-cover"
                />
              </div>
              <span className="text-xs font-mono text-white whitespace-nowrap">
                {image.timestamp}
              </span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}