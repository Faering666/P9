'use client';

import { useState, useEffect } from 'react';

interface ImageData {
  filename: string;
  timestamp: string;
}

interface SatelliteViewerProps {
  imageFolder?: string;
}

export default function SatelliteViewer({
  imageFolder = '/satellite-images'
}: SatelliteViewerProps) {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [images, setImages] = useState<ImageData[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Fetch the images list from the JSON file
    fetch(`${imageFolder}/images.json`)
      .then(res => res.json())
      .then((data: ImageData[]) => {
        setImages(data);
        setLoading(false);
      })
      .catch(error => {
        console.error('Error loading images:', error);
        setLoading(false);
      });
  }, [imageFolder]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <p className="text-lg text-gray-600 dark:text-gray-400">Loading images...</p>
      </div>
    );
  }

  if (images.length === 0) {
    return (
      <div className="flex items-center justify-center h-64">
        <p className="text-lg text-gray-600 dark:text-gray-400">No images found</p>
      </div>
    );
  }

  const currentImage = images[currentIndex];
  const imagePath = `${imageFolder}/${currentImage.filename}`;

  return (
    <div className="flex flex-col items-center gap-6 w-full max-w-6xl mx-auto p-6">
      {/* Main Image Display */}
      <div className="relative w-full aspect-[4/3] bg-gray-100 dark:bg-gray-800 rounded-lg overflow-hidden shadow-lg">
        <img
          src={imagePath}
          alt={`Satellite image at ${currentImage.timestamp}`}
          className="w-full h-full object-contain"
        />

        {/* Image Info Overlay */}
        <div className="absolute top-4 left-4 bg-black/70 text-white px-4 py-2 rounded-lg">
          <p className="text-sm font-mono">{currentImage.timestamp}</p>
        </div>
      </div>

      {/* Thumbnail Timeline */}
      <div className="w-full">
        <h2 className="text-center font-medium text-lg mb-4">Select Time</h2>
        <div className="flex gap-3 overflow-x-auto pb-2 px-2">
          {images.map((image, index) => {
            const thumbPath = `${imageFolder}/${image.filename}`;
            return (
              <button
                key={index}
                onClick={() => setCurrentIndex(index)}
                className={`flex-shrink-0 flex flex-col items-center gap-2 p-2 rounded-lg transition-all ${
                  currentIndex === index
                    ? 'bg-blue-500 shadow-lg scale-105'
                    : 'bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600'
                }`}
              >
                {/* Thumbnail */}
                <div className="relative w-32 h-24 bg-gray-100 dark:bg-gray-800 rounded overflow-hidden">
                  <img
                    src={thumbPath}
                    alt={`Thumbnail ${index + 1}`}
                    className="w-full h-full object-cover"
                  />
                </div>

                {/* Timestamp */}
                <span className={`text-xs font-mono whitespace-nowrap ${
                  currentIndex === index
                    ? 'text-white font-semibold'
                    : 'text-gray-700 dark:text-gray-300'
                }`}>
                  {image.timestamp}
                </span>
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}