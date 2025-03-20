import React, { useState, useRef, useEffect } from "react";
import * as ort from "onnxruntime-web";

const CatDogClassifier = () => {
    const [result, setResult] = useState(null);
    const canvasRef = useRef(null);
    const sessionRef = useRef(null);

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

    // Handle file upload
    const handleFileUpload = (event) => {
        const file = event.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                const img = new Image();
                img.src = e.target.result;
                img.onload = () => processImage(img);
            };
            reader.readAsDataURL(file);
        }
    };

    // Process uploaded image
    const processImage = (image) => {
        const canvas = canvasRef.current;
        const context = canvas.getContext("2d");
        canvas.width = 64;  // Fix input size to match model
        canvas.height = 64;
        context.drawImage(image, 0, 0, 64, 64);
        classifyImage(canvas);
    };
    
    

    // Convert image data into a tensor for ONNX
    const preprocessImage = (canvas) => {
        const ctx = canvas.getContext("2d");
        const imageData = ctx.getImageData(0, 0, 64, 64).data;  // Use 64x64
        let floatArray = new Float32Array(3 * 64 * 64);  // Correct size
    
        for (let i = 0; i < 64 * 64; i++) {
            floatArray[i] = imageData[i * 4] / 255.0;          // Red channel
            floatArray[i + 4096] = imageData[i * 4 + 1] / 255.0;  // Green channel
            floatArray[i + 8192] = imageData[i * 4 + 2] / 255.0;  // Blue channel
        }
    
        return new ort.Tensor("float32", floatArray, [1, 3, 64, 64]);  // Correct shape
    };
    
    

    // Run ONNX inference
    const classifyImage = async (canvas) => {
        if (!sessionRef.current) {
            console.error("❌ Model not loaded yet.");
            return;
        }

        try {
            const inputTensor = preprocessImage(canvas);
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
            <input type="file" accept="image/*" onChange={handleFileUpload} />
            <canvas ref={canvasRef} style={{ display: "none" }}></canvas>
            <br />
            {result && <h2>{result}</h2>}
        </div>
    );
};

export default CatDogClassifier;