import React, { useRef, useState, useEffect } from 'react';
import Webcam from 'react-webcam';
import axios from 'axios';
import './App.css';

const App = () => {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const [dominantEmotion, setDominantEmotion] = useState(null);

  useEffect(() => {
    const intervalId = setInterval(capture, 1000); // Capture frame every second
    return () => clearInterval(intervalId); // Cleanup on component unmount
  }, []);

  const capture = async () => {
    if (webcamRef.current && canvasRef.current) {
      const imageSrc = webcamRef.current.getScreenshot();
      try {
        const response = await axios.post('http://localhost:5000/detect_emotion', {
          image: imageSrc.split(',')[1],  // remove data:image/jpeg;base64,
        });

        if (response.data.error) {
          console.error(response.data.error);
        } else {
          const emotions = response.data;
          const dominant = Object.keys(emotions).reduce((a, b) => emotions[a] > emotions[b] ? a : b);
          setDominantEmotion(dominant);

          // Draw green box around the face
          const context = canvasRef.current.getContext('2d');
          const { width, height } = webcamRef.current.video;

          canvasRef.current.width = width;
          canvasRef.current.height = height;

          context.clearRect(0, 0, width, height);
          context.strokeStyle = 'green';
          context.lineWidth = 3;

          // Assuming the face occupies the central portion for simplicity
          const faceX = width * 0.25;
          const faceY = height * 0.25;
          const faceWidth = width * 0.5;
          const faceHeight = height * 0.5;

          context.strokeRect(faceX, faceY, faceWidth, faceHeight);
        }
      } catch (error) {
        console.error('An error occurred while detecting emotion.', error);
      }
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Real-time Emotion Detector</h1>
        <div style={{ position: 'relative' }}>
          <Webcam
            audio={false}
            ref={webcamRef}
            screenshotFormat="image/jpeg"
          />
          <canvas
            ref={canvasRef}
            style={{ position: 'absolute', top: 0, left: 0 }}
          />
        </div>
        {dominantEmotion && (
          <div>
            <h2>Dominant Emotion: {dominantEmotion}</h2>
          </div>
        )}
      </header>
    </div>
  );
};

export default App;
