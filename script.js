const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

// Ensure proper video size
canvas.width = 640;
canvas.height = 480;

// Load MediaPipe Hands
const hands = new Hands({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
});

hands.setOptions({
    maxNumHands: 1,
    minDetectionConfidence: 0.7,
    minTrackingConfidence: 0.7
});

console.log(hands); // ✅ Check if it's loading correctly


// Function to process results
hands.onResults(results => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    if (results.multiHandLandmarks) {
        for (const landmarks of results.multiHandLandmarks) {
            drawLandmarks(ctx, landmarks);
        }
    }
});

// Function to draw landmarks
function drawLandmarks(ctx, landmarks) {
    ctx.strokeStyle = "red";
    ctx.lineWidth = 2;

    for (let i = 0; i < landmarks.length; i++) {
        let x = landmarks[i].x * canvas.width;
        let y = landmarks[i].y * canvas.height;
        ctx.beginPath();
        ctx.arc(x, y, 5, 0, 2 * Math.PI);
        ctx.fillStyle = "blue";
        ctx.fill();
    }
}

// Setup Camera
const camera = new Camera(video, {
    onFrame: async () => {
        await hands.send({image: video});
    },
    width: 640,
    height: 480
});
camera.start();
