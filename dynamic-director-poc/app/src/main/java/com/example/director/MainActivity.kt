package com.example.director

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import org.opencv.android.OpenCVLoader
import java.util.concurrent.Executors

class MainActivity : ComponentActivity() {

    private val cameraPermissionRequest = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            Log.d("DynamicDirector", "Camera permission granted")
            recreate()
        } else {
            Log.e("DynamicDirector", "Camera permission denied")
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        // Initialize OpenCV
        if (OpenCVLoader.initDebug()) {
            Log.d("DynamicDirector", "OpenCV loaded successfully")
        } else {
            Log.e("DynamicDirector", "OpenCV initialization failed")
        }

        val hasPermission = ContextCompat.checkSelfPermission(
            this,
            Manifest.permission.CAMERA
        ) == PackageManager.PERMISSION_GRANTED

        if (!hasPermission) {
            cameraPermissionRequest.launch(Manifest.permission.CAMERA)
        }

        setContent {
            MaterialTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    CameraScreen(hasPermission)
                }
            }
        }
    }
}

@Composable
fun CameraScreen(hasPermission: Boolean) {
    if (hasPermission) {
        val context = LocalContext.current
        val lifecycleOwner = LocalLifecycleOwner.current
        
        var currentScore by remember { mutableStateOf(0f) }
        var detectionStatus by remember { mutableStateOf("Waiting for AI...") }
        var currentFace by remember { mutableStateOf<FaceResult?>(null) }
        var currentPose by remember { mutableStateOf<PoseResult?>(null) }
        
        // Initialize AI models
        val faceAnalyzer = remember { FaceAnalyzer() }
        val poseAnalyzer = remember { PoseAnalyzer() }
        val config = remember { ConfigManager.loadConfig(context) }

        Box(modifier = Modifier.fillMaxSize()) {
            AndroidView(
                factory = { ctx ->
                    val previewView = PreviewView(ctx)
                    val cameraProviderFuture = ProcessCameraProvider.getInstance(ctx)
                    
                    cameraProviderFuture.addListener({
                        val cameraProvider = cameraProviderFuture.get()
                        
                        val preview = Preview.Builder().build().also {
                            it.setSurfaceProvider(previewView.surfaceProvider)
                        }
                        
                        val imageAnalysis = ImageAnalysis.Builder()
                            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                            .build()
                            
                        val imageCapture = ImageCapture.Builder()
                            .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                            .build()
                            
                        // Phase 5: Auto Shutter State variables
                        var consecutiveHighScores = 0
                        var lastCaptureTime = 0L
                        val cooldownMs = 3000L
                        var isTakingPhoto = false

                        imageAnalysis.setAnalyzer(Executors.newSingleThreadExecutor()) { imageProxy ->
                            try {
                                val rawBitmap = imageProxy.toBitmap()
                                
                                // Rotate bitmap to be upright (Object Detection fails if image is sideways)
                                val matrix = android.graphics.Matrix()
                                matrix.postRotate(imageProxy.imageInfo.rotationDegrees.toFloat())
                                val bitmap = android.graphics.Bitmap.createBitmap(
                                    rawBitmap, 0, 0, rawBitmap.width, rawBitmap.height, matrix, true
                                )
                                
                                // Chain Analyzers: Face -> Pose
                                faceAnalyzer.analyze(bitmap, { faceResult ->
                                    currentFace = faceResult
                                    
                                    poseAnalyzer.analyze(bitmap, { poseResult ->
                                        currentPose = poseResult
                                        
                                        val faceScore = faceResult?.let { ScoringEngine.calculateFaceScore(it) } ?: 0f
                                        val poseScore = poseResult?.score ?: 0f
                                        val score = maxOf(faceScore, poseScore)
                                        currentScore = score
                                        
                                        // Phase 6 & 7: Auto Shutter Logic
                                        val now = System.currentTimeMillis()
                                        if (isTakingPhoto) {
                                            detectionStatus = "📸 Taking Photo..."
                                        } else if (now - lastCaptureTime < cooldownMs) {
                                            detectionStatus = "❄️ Cooldown..."
                                            consecutiveHighScores = 0
                                        } else {
                                            if (poseResult != null && poseResult.actionName != null) {
                                                detectionStatus = "Action: ${poseResult.actionName}!"
                                            } else if (faceResult != null) {
                                                val smilePct = (faceResult.smilingProbability * 100).toInt()
                                                detectionStatus = "Face Detected (Smile: $smilePct%)"
                                            } else {
                                                detectionStatus = "Searching..."
                                            }
                                            
                                            // Shutter threshold (Face > 80 or Pose = 100)
                                            if (score >= 80f) {
                                                consecutiveHighScores++
                                                if (consecutiveHighScores >= 3) {
                                                    isTakingPhoto = true
                                                    lastCaptureTime = now
                                                    consecutiveHighScores = 0
                                                    
                                                    val file = java.io.File(ctx.filesDir, "director_capture_${now}.jpg")
                                                    val outputOptions = ImageCapture.OutputFileOptions.Builder(file).build()
                                                    
                                                    imageCapture.takePicture(
                                                        outputOptions,
                                                        ContextCompat.getMainExecutor(ctx),
                                                        object : ImageCapture.OnImageSavedCallback {
                                                            override fun onImageSaved(output: ImageCapture.OutputFileResults) {
                                                                isTakingPhoto = false
                                                                Log.d("DynamicDirector", "Photo saved to ${file.absolutePath}")
                                                            }
                                                            override fun onError(exc: ImageCaptureException) {
                                                                isTakingPhoto = false
                                                                Log.e("DynamicDirector", "Capture failed", exc)
                                                            }
                                                        }
                                                    )
                                                }
                                            } else {
                                                consecutiveHighScores = 0
                                            }
                                        }
                                        imageProxy.close()
                                    }, {
                                        // Pose Complete
                                    })
                                }, {
                                    // Face Complete
                                })
                            } catch (e: Exception) {
                                Log.e("DynamicDirector", "Analysis failed", e)
                                imageProxy.close()
                            }
                        }
                        
                        val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
                        
                        try {
                            cameraProvider.unbindAll()
                            cameraProvider.bindToLifecycle(
                                lifecycleOwner,
                                cameraSelector,
                                preview,
                                imageAnalysis,
                                imageCapture
                            )
                        } catch (e: Exception) {
                            Log.e("DynamicDirector", "Use case binding failed", e)
                        }
                    }, ContextCompat.getMainExecutor(ctx))
                    
                    previewView
                },
                modifier = Modifier.fillMaxSize()
            )
            
            // Developer Dashboard Overlay
            val appVersion = "v0.3.1"
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(16.dp),
                contentAlignment = Alignment.TopStart
            ) {
                val faceText = currentFace?.let {
                    val smile = (it.smilingProbability * 100).toInt()
                    val isLooking = Math.abs(it.headEulerAngleY) < 15f && Math.abs(it.headEulerAngleZ) < 15f
                    "Smile: $smile%\nLooking at Camera: ${if (isLooking) "Yes" else "No"}"
                } ?: "Smile: -\nLooking at Camera: -"
                
                val poseText = "Pose: ${currentPose?.actionName ?: "None"}"
                
                Text(
                    text = "Dynamic Director $appVersion\nStatus: $detectionStatus\nScore: ${currentScore.toInt()}%\n$poseText\n$faceText",
                    color = if (currentScore > 85f) Color.Yellow else Color.Green,
                    style = MaterialTheme.typography.bodyLarge
                )
            }
            
            // Draw BBox Overlay
            currentFace?.let { face ->
                androidx.compose.foundation.Canvas(modifier = Modifier.fillMaxSize()) {
                    val w = size.width
                    val h = size.height
                    val left = face.x1 * w
                    val top = face.y1 * h
                    val right = face.x2 * w
                    val bottom = face.y2 * h
                    
                    drawRect(
                        color = Color.Green,
                        topLeft = androidx.compose.ui.geometry.Offset(left, top),
                        size = androidx.compose.ui.geometry.Size(right - left, bottom - top),
                        style = androidx.compose.ui.graphics.drawscope.Stroke(width = 5f)
                    )
                }
            }

            // Draw Pose Landmarks Overlay
            currentPose?.let { pose ->
                androidx.compose.foundation.Canvas(modifier = Modifier.fillMaxSize()) {
                    val w = size.width
                    val h = size.height
                    for (point in pose.landmarks) {
                        drawCircle(
                            color = Color.Cyan,
                            radius = 8f,
                            center = androidx.compose.ui.geometry.Offset(point.x * w, point.y * h)
                        )
                    }
                }
            }
        }
    } else {
        Box(
            modifier = Modifier.fillMaxSize(),
            contentAlignment = Alignment.Center
        ) {
            Text("Camera permission is required to use this app.")
        }
    }
}
