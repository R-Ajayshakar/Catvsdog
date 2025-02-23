import React, { useState, useRef, useEffect } from "react";
import * as ort from "onnxruntime-web";
import * as tf from "@tensorflow/tfjs";

const CatDogClassifier = () => {
  const [result, setResult] = useState(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const sessionRef = useRef(null); // Store model session

  // Load the ONNX model
  useEffect(() => {
    async function loadModel() {
      try {
        sessionRef.current = await ort.InferenceSession.create("catvsdog.onnx");
        console.log("Model loaded successfully!");
      } catch (error) {
        console.error("Failed to load model:", error);
      }
    }
    loadModel();
  }, []);

  // Start Webcam
  useEffect(() => {
    const startCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        videoRef.current.srcObject = stream;
      } catch (error) {
        console.error("Error accessing webcam:", error);
      }
    };
    startCamera();
  }, []);

  // Capture image from webcam
  const captureImage = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const context = canvas.getContext("2d");

    canvas.width = 224;
    canvas.height = 224;
    context.drawImage(video, 0, 0, 224, 224);

    return tf.browser.fromPixels(canvas).toFloat().div(tf.scalar(255.0)).expandDims();
  };

  // Run ONNX inference
  const classifyImage = async () => {
    if (!sessionRef.current) {
      console.error("Model not loaded yet.");
      return;
    }

    try {
      const inputTensor = captureImage();
      const feeds = { [sessionRef.current.inputNames[0]]: inputTensor };
      const outputData = await sessionRef.current.run(feeds);
      const predictions = outputData[sessionRef.current.outputNames[0]].data;

      // Apply softmax
      const expScores = predictions.map(Math.exp);
      const sumExpScores = expScores.reduce((a, b) => a + b, 0);
      const probabilities = expScores.map((score) => score / sumExpScores);

      // Determine if it's a cat or dog
      setResult(probabilities[0] > probabilities[1] ? "It's a Cat! 🐱" : "It's a Dog! 🐶");
    } catch (error) {
      console.error("Error during inference:", error);
    }
  };

  return (
    <div style={{ textAlign: "center" }}>
      <h1>Cat vs Dog Classifier</h1>
      <video ref={videoRef} autoPlay playsInline width="320" height="240"></video>
      <canvas ref={canvasRef} style={{ display: "none" }}></canvas>
      <br />
      <button onClick={classifyImage}>Capture & Predict</button>
      {result && <h2>{result}</h2>}
    </div>
  );
};

export default CatDogClassifier;
