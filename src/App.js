import React, { useRef, useEffect, useState } from "react";
import * as tf from "@tensorflow/tfjs";
import "./index.css";

function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [model, setModel] = useState(null);

  useEffect(() => {
    async function setupWebcam() {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      videoRef.current.srcObject = stream;
    }

    async function loadModel() {
      const loadedModel = await tf.loadLayersModel("/asl_model/model.json");
      setModel(loadedModel);
    }

    setupWebcam();
    loadModel();
  }, []);

  const handleCapture = () => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
    let image = tf.browser.fromPixels(canvas);

    // Preprocess the image
    image = image.resizeNearestNeighbor([100, 100]); // Resize to 255x255 pixels
    image = image.toFloat().div(tf.scalar(100.0)); // Normalize to [0, 1]
    image = image.expandDims(0); // For batch input

    // Get prediction
    if (model) {
      const prediction = model.predict(image);
      const predictedClass = prediction.argMax(1).dataSync()[0];
      // For this example, let's assume that 0 corresponds to 'A', 1 to 'B', and so on.
      const alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
      const predictedLetter = alphabet[predictedClass];

      alert(`Predicted sign: ${predictedLetter}`);
    }
  };

  return (
    <div className="app-container">
      <div className="camera-container">
        <img
          src="./showAndTellLogo.png"
          alt="Show and Tell Logo"
          className="logo"
        />
        <video
          ref={videoRef}
          autoPlay
          playsInline
          width="400"
          height="400"
        ></video>
        <canvas ref={canvasRef} width="400" height="400"></canvas>
        <button onClick={handleCapture}>Capture</button>
      </div>
    </div>
  );
}

export default App;
