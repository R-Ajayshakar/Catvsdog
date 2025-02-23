import React, { useState, useRef, useEffect } from "react";
import * as ort from "onnxruntime-web";

const CatDogClassifier = () => {
    const [result, setResult] = useState(null);
    const [facingMode, setFacingMode] = useState("user"); // "user" = front, "environment" = back
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const sessionRef = useRef(null);
    let streamRef = useRef(null); // Store the current stream

    // Load the ONNX model
    useEffect(() => {
        const loadModel = async () => {
            try {
                sessionRef.current = await ort.InferenceSession.create("/catvsdog.onnx");
                console.log("✅ ONNX Model Loaded");
            } catch (error) {
                console.error("❌ Error loading ONNX model:", error);
            }
        };
        loadModel();
    }, []);

    // Start Webcam
    const startCamera = async () => {
        try {
            // Stop the previous stream if it exists
            if (streamRef.current) {
                streamRef.current.getTracks().forEach((track) => track.stop());
            }

            // Start a new camera stream with the selected facing mode
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode },
            });

            // Set the new stream
            streamRef.current = stream;
            videoRef.current.srcObject = stream;
        } catch (error) {
            console.error("❌ Error accessing webcam:", error);
        }
    };

    useEffect(() => {
        startCamera();
    }, [facingMode]); // Restart camera when facingMode changes

    // Toggle between front and back cameras
    const toggleCamera = () => {
        setFacingMode((prev) => (prev === "user" ? "environment" : "user"));
    };

    // Capture image from webcam
    const captureImage = () => {
        const video = videoRef.current;
        const canvas = canvasRef.current;
        const context = canvas.getContext("2d");

        canvas.width = 32;
        canvas.height = 32;
        context.drawImage(video, 0, 0, 32, 32);

        return preprocessImage(canvas);
    };

    // Convert image data into a tensor for ONNX
    const preprocessImage = (canvas) => {
        const ctx = canvas.getContext("2d");
        const imageData = ctx.getImageData(0, 0, 32, 32).data;
        let floatArray = new Float32Array(3 * 32 * 32);

        for (let i = 0; i < 32 * 32; i++) {
            floatArray[i] = imageData[i * 4] / 255.0;
            floatArray[i + 1024] = imageData[i * 4 + 1] / 255.0;
            floatArray[i + 2048] = imageData[i * 4 + 2] / 255.0;
        }

        return new ort.Tensor("float32", floatArray, [1, 3, 32, 32]);
    };

    // Run ONNX inference
    const classifyImage = async () => {
        if (!sessionRef.current) {
            console.error("❌ Model not loaded yet.");
            return;
        }

        try {
            const inputTensor = captureImage();
            const feeds = { [sessionRef.current.inputNames[0]]: inputTensor };
            const outputData = await sessionRef.current.run(feeds);
            const predictions = outputData[sessionRef.current.outputNames[0]].data;

            // Apply softmax to get probabilities
            const expScores = predictions.map(Math.exp);
            const sumExpScores = expScores.reduce((a, b) => a + b, 0);
            const probabilities = expScores.map((score) => score / sumExpScores);

            console.log("🐶🐱 Model Predictions:", probabilities);

            if (probabilities[0] > 0.9 || probabilities[1] > 0.9) {
                setResult(probabilities[0] > probabilities[1] ? "It's a Cat! 🐱" : "It's a Dog! 🐶");
            } else {
                setResult("Not a Cat or Dog ❌");
            }
        } catch (error) {
            console.error("❌ Error during inference:", error);
        }
    };

    return (
        <div style={{ textAlign: "center" }}>
            <h1>Cat vs Dog Classifier</h1>
            <video ref={videoRef} autoPlay playsInline width="320" height="240"></video>
            <canvas ref={canvasRef} style={{ display: "none" }}></canvas>
            <br />
            <button onClick={classifyImage}>Capture & Predict</button>
            <button onClick={toggleCamera}>Flip Camera</button>
            {result && <h2>{result}</h2>}
        </div>
    );
};

export default CatDogClassifier;
