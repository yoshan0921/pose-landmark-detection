import './App.css';
import { useEffect, useRef, useState } from 'react';
import { PoseLandmarker, FilesetResolver, DrawingUtils } from 'https://cdn.skypack.dev/@mediapipe/tasks-vision@0.10.0';
import { Box, Button, Stack, Typography, createTheme, ThemeProvider, CssBaseline } from '@mui/material';
import { DataGrid } from '@mui/x-data-grid';

const landmarkNames = [
  'NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER', 'RIGHT_EYE_INNER',
  'RIGHT_EYE', 'RIGHT_EYE_OUTER', 'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 'MOUTH_RIGHT',
  'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST',
  'LEFT_PINKY', 'RIGHT_PINKY', 'LEFT_INDEX', 'RIGHT_INDEX', 'LEFT_THUMB', 'RIGHT_THUMB',
  'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE', 
  'LEFT_HEEL', 'RIGHT_HEEL', 'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX'
];

// Light gray theme for MUI
const lightGrayTheme = createTheme({
  palette: {
    mode: 'light',
    background: {
      default: '#f5f5f5',
      paper: '#ffffff',
    },
    text: {
      primary: '#333333',
    },
  },
  typography: {
    fontSize: 12,
  },
});

function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const poseLandmarkerRef = useRef(null);
  const webcamRunningRef = useRef(false);
  const [webcamRunning, setWebcamRunning] = useState(false);
  const runningMode = useRef('IMAGE');
  const [landmarkData, setLandmarkData] = useState([]); 
  const lastUpdateRef = useRef(0); 

  // Setup MediaPipe PoseLandmarker when component mounts
  useEffect(() => {
    const createPoseLandmarker = async () => {
      try {
        const vision = await FilesetResolver.forVisionTasks(
          'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm'
        );
        const poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
            delegate: 'GPU',
          },
          runningMode: runningMode.current,
          numPoses: 2,
        });
        poseLandmarkerRef.current = poseLandmarker;
      } catch (error) {
        console.error("Error loading PoseLandmarker:", error);
      }
    };

    createPoseLandmarker();
  }, []);

  const enableCam = async () => {
    if (!poseLandmarkerRef.current) {
      console.error('PoseLandmarker not loaded.');
      return;
    }

    const videoElement = videoRef.current;
    const canvasElement = canvasRef.current;
    const canvasCtx = canvasElement.getContext('2d');

    // if (webcamRunningRef.current) {
    if (webcamRunning) {
      // Stop webcam
      const stream = videoElement.srcObject;
      const tracks = stream.getTracks();
      tracks.forEach(track => track.stop());
      videoElement.srcObject = null;
      webcamRunningRef.current = false;
      setWebcamRunning(false); // Update state
      console.log("Webcam disabled");

      canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    } else {
      // Start webcam
      const constraints = { video: true };

      try {
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        videoElement.srcObject = stream;
        videoElement.addEventListener('loadeddata', () => {
          console.log("Webcam video loaded:", videoElement.videoWidth, videoElement.videoHeight);
          webcamRunningRef.current = true;
          setWebcamRunning(true); // Update state
          // predictWebcam(); // Start prediction
        });
      } catch (error) {
        console.error("Error accessing the webcam: ", error);
      }
    }
  };

  // Add useEffect to track webcamRunning changes
  useEffect(() => {
    if (webcamRunning) {
      predictWebcam(); // Start prediction when webcam is running
    }
  }, [webcamRunning]); // This effect runs when webcamRunning changes

  const predictWebcam = async () => {
    const videoElement = videoRef.current;
    const canvasElement = canvasRef.current;
    const canvasCtx = canvasElement.getContext('2d');
    const drawingUtils = new DrawingUtils(canvasCtx);
  
    // Check if video element is available
    if (videoElement.videoWidth === 0 || videoElement.videoHeight === 0) {
      console.log("Video element not available or has been stopped.");
      return;
    }
  
    // Adjust canvas size to match video
    canvasElement.width = videoElement.videoWidth;
    canvasElement.height = videoElement.videoHeight;
  
    if (runningMode.current === 'IMAGE') {
      runningMode.current = 'VIDEO';
      await poseLandmarkerRef.current.setOptions({ runningMode: 'VIDEO' });
    }
  
    const startTimeMs = performance.now();
  
    try {
      const results = await poseLandmarkerRef.current.detectForVideo(videoElement, startTimeMs);
  
      canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  
      if (results && results.landmarks && results.landmarks.length > 0) {
        const updatedLandmarkData = results.landmarks[0].map((landmark, index) => ({
          id: index,
          name: landmarkNames[index],
          x: landmark?.x?.toFixed(2) ?? 'NA',
          y: landmark?.y?.toFixed(2) ?? 'NA',
          z: landmark?.z?.toFixed(2) ?? 'NA',
        }));

        const now = performance.now();
        if (now - lastUpdateRef.current > 500) {
          lastUpdateRef.current = now;
          setLandmarkData(updatedLandmarkData);
        }

        drawingUtils.drawConnectors(results.landmarks[0], PoseLandmarker.POSE_CONNECTIONS);

        drawingUtils.drawLandmarks(results.landmarks[0], {
          radius: (data) => {
            // console.log(data.index);
            return DrawingUtils.lerp(data.from.z, -0.15, 0.1, 5, 1);
          },
          color: (data) => {
            return data.index === 0 ? 'green' : 'green';
          }
        });

        results.landmarks[0].forEach((landmark, index) => {
          const x = landmark.x * canvasElement.width;
          const y = landmark.y * canvasElement.height;
        
          canvasCtx.font = '10px Roboto';
          canvasCtx.fillStyle = 'blue';
          canvasCtx.fontWeight = 'bold';
          canvasCtx.fillText(`ID ${index}`, x, y);
        });
      }
    } catch (error) {
      console.error("Error during pose detection:", error);
    }
  
    // Continue predictions as long as webcam is running
    // if (webcamRunningRef.current) {
    console.log("Webcam running:", webcamRunning);
    if (webcamRunning) {
      window.requestAnimationFrame(predictWebcam);
    }
  };

  const columns = [
    { field: 'id', headerName: 'ID', width: 30 }, 
    { field: 'name', headerName: 'Landmark', minWidth: 140 },
    { field: 'x', headerName: 'X', width: 70 },
    { field: 'y', headerName: 'Y', width: 70 },
    { field: 'z', headerName: 'Z', width: 70 },
  ];

  return (
    <ThemeProvider theme={lightGrayTheme}>
      <CssBaseline />
      <Stack sx={{ height: '100vh', backgroundColor: 'background.default' }} spacing={2}>
        <Stack direction="row" sx={{ flex: 1, overflow: 'hidden' }}>
          <Box sx={{ flexBasis: '70%', position: 'relative', overflow: 'hidden', backgroundColor: 'background.paper' }}>
            <video
              ref={videoRef}
              style={{ width: '100%', height: '100%', objectFit: 'cover', position: 'absolute', top: 0, left: 0 }}
              autoPlay
              playsInline
            ></video>
            <canvas
              ref={canvasRef}
              style={{ width: '100%', height: '100%', objectFit: 'cover', position: 'absolute', top: 0, left: 0 }}
            ></canvas>
          </Box>
          <Box sx={{ flexBasis: '30%', padding: 1, overflowY: 'auto', maxHeight: '100%', backgroundColor: 'background.paper' }}>
            <Typography variant="h5" sx={{ marginBottom: 1 }}>Pose Landmarks</Typography>
            <div style={{ height: 'calc(100% - 50px)', width: '100%' }}>
            <DataGrid
              rows={landmarkData} 
              columns={columns} 
              pageSize={landmarkNames.length} 
              // checkboxSelection
              disableColumnMenu 
              hideFooter
              sx={{ 
                '& .MuiDataGrid-cell': { 
                  padding: '4px',
                  fontSize: '0.8rem',
                } 
              }} 
            />
            </div>
          </Box>
        </Stack>
        <Box
          sx={{
            padding: 2,
            backgroundColor: 'background.paper',
            textAlign: 'center',
            borderTop: 1,
            borderTopColor: 'divider',
            borderTopStyle: 'solid',
          }}
        >
          <Button variant="contained" onClick={enableCam}>
            {/* {webcamRunningRef.current ? 'DISABLE WEBCAM' : 'ENABLE WEBCAM'} */}
            {webcamRunning ? 'DISABLE WEBCAM' : 'ENABLE WEBCAM'}
          </Button>
        </Box>
      </Stack>
    </ThemeProvider>
  );
}

export default App;
