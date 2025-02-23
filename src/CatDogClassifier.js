import React, { useState, useRef, useEffect } from "react";
import * as ort from "onnxruntime-web";

const CatDogClassifier = () => {
    const [result, setResult] = useState(null);
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    let session = null;

    // Load the ONNX model
    useEffect(() => {
        const loadModel = async () => {
            session = await ort.InferenceSession.create("/catvsdog.onnx");
            console.log("ONNX Model Loaded");
        };
        loadModel();
    }, []);

    // Start Webcam
    useEffect(() => {
        const startCamera = async () => {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            videoRef.current.srcObject = stream;
        };
        startCamera();
    }, []);

    // Capture image from webcam
    const captureImage = () => {
        const video = videoRef.current;
        const canvas = canvasRef.current;
        const context = canvas.getContext("2d");

        canvas.width = 32;
        canvas.height = 32;
        context.drawImage(video, 0, 0, 32, 32);

        // Convert to tensor
        const imageData = context.getImageData(0, 0, 32, 32).data;
        return preprocessImage(imageData);
    };

    // Convert image data into a tensor for ONNX
    const preprocessImage = (imageData) => {
        let floatArray = new Float32Array(3 * 32 * 32);

        for (let i = 0; i < 32 * 32; i++) {
            floatArray[i] = imageData[i * 4] / 255.0;       // Red
            floatArray[i + 1024] = imageData[i * 4 + 1] / 255.0; // Green
            floatArray[i + 2048] = imageData[i * 4 + 2] / 255.0; // Blue
        }

        return new ort.Tensor("float32", floatArray, [1, 3, 32, 32]);
    };

    // Run ONNX inference
    const classifyImage = async () => {
        if (!session) {
            console.error("Model not loaded yet.");
            return;
        }

        const inputTensor = captureImage();
        const feeds = { [session.inputNames[0]]: inputTensor };
        const outputData = await session.run(feeds);
        const predictions = outputData[session.outputNames[0]].data;

        // Apply softmax to get probabilities
        const expScores = predictions.map(Math.exp);
        const sumExpScores = expScores.reduce((a, b) => a + b, 0);
        const probabilities = expScores.map((score) => score / sumExpScores);

        // Determine if it's a cat or dog
        setResult(probabilities[0] > probabilities[1] ? "It's a Cat! 🐱" : "It's a Dog! 🐶");
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
