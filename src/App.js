import React, { useRef, useEffect, useState } from 'react';
import * as tf from '@tensorflow/tfjs';

function App() {
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const [model, setModel] = useState(null);

    useEffect(() => {
        async function setupWebcam() {
            const stream = await navigator.mediaDevices.getUserMedia({ 'video': true });
            videoRef.current.srcObject = stream;
        }

        async function loadModel() {
            const loadedModel = await tf.loadLayersModel('../asl_model/model.json');
            setModel(loadedModel);
        }

        setupWebcam();
        loadModel();
    }, []);


    const code = {
        0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
        10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
        20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'del', 27: 'nothing', 28: 'space'
      };
      
      function getLetterByNumber(number) {
        // Check if the number is a valid key in the code dictionary
        if (number >= 0 && number <= 28) {
          return code[number];
        } else {
          return "Invalid number"; // Handle invalid numbers
        }
      }

    const handleCapture = () => {
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
        let image = tf.browser.fromPixels(canvas);

        // Preprocess the image
        image = image.resizeNearestNeighbor([128, 128]);  // Resize to 255x255 pixels
        image = image.toFloat().div(tf.scalar(255.0));   // Normalize to [0, 1]
        image = image.expandDims(0);  // For batch input

        // Get prediction
        if (model) {
            const prediction = model.predict(image);
            console.log(prediction);
            const predictedClass = prediction.argMax(1).dataSync()[0];
            // For this example, let's assume that 0 corresponds to 'A', 1 to 'B', and so on.
            //  const alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
            const predictedLetter = getLetterByNumber(predictedClass);

            alert(`Predicted sign: ${predictedLetter}`);
        }
    };

    return (
        <div>
            <video ref={videoRef} autoPlay playsInline width="640" height="480"></video>
            <canvas ref={canvasRef} width="640" height="480"></canvas>
            <button onClick={handleCapture}>Capture</button>
        </div>
    );
}

export default App;
