import React, { useRef, useEffect, useState } from "react";
import * as tf from "@tensorflow/tfjs";
import "./index.css";

import Button from "@material-ui/core/Button";
import {Dialog, DialogTitle, DialogContent, DialogActions } from '@material-ui/core';
import CameraIcon from "@material-ui/icons/PhotoCamera"; // Optional, if you want an icon on the button

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

  const [openDialog, setOpenDialog] = React.useState(false); // State to control the dialog open/close
  const [predictedLetter, setPredictedLetter] = React.useState(""); // State to store the predicted letter

  const handleCapture = () => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    // ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
    // ctx.clearRect(0, 0, canvas.width, canvas.height);
    const videoWidth = videoRef.current.videoWidth;
    const videoHeight = videoRef.current.videoHeight;

    const aspectRatio = videoWidth / videoHeight;

    let targetWidth = canvas.width;
    let targetHeight = targetWidth / aspectRatio;

    // if height gets too big, adjust width instead
    if (targetHeight > canvas.height) {
      targetHeight = canvas.height;
      targetWidth = targetHeight * aspectRatio;
    }
    ctx.drawImage(
      videoRef.current,
      (canvas.width - targetWidth) / 2,
      (canvas.height - targetHeight) / 2,
      targetWidth,
      targetHeight
    );
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

      setPredictedLetter(predictedLetter);
      setOpenDialog(true); // Open the dialog
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
        <Button
          variant="contained"
          style={{
            backgroundColor: "#EC255A",
            color: "white",
            marginBottom: "20px",
          }}
          startIcon={<CameraIcon />} // Optional, if you want an icon on the button
          onClick={handleCapture}
        >
          Capture
        </Button>
         Material-UI Dialog
        <Dialog 
            fullWidth
            open={openDialog} onClose={() => setOpenDialog(false)}
        >
            <DialogTitle 
            >Predicted Sign</DialogTitle>
            <DialogContent>
                <p> Predicted sign: {predictedLetter}</p>
            </DialogContent>
            <DialogActions>
            <Button onClick={() => setOpenDialog(false)}>
                Close
            </Button>
            </DialogActions>
        </Dialog>
      </div>
    </div>
  );
}

export default App;
