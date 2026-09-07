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
        var detectionStatus by remember { mutableStateOf("Waiting for Detection...") }
        
        // Initialize AI models and configs once
        val yoloDetector = remember { YoloDetector(context) }
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
                            
                        imageAnalysis.setAnalyzer(Executors.newSingleThreadExecutor()) { imageProxy ->
                            try {
                                val bitmap = imageProxy.toBitmap()
                                
                                // 1. TFLite Detection
                                val detection = yoloDetector.detect(bitmap)
                                
                                if (detection != null && config != null) {
                                    // 2. OpenCV Scoring
                                    val score = ScoringEngine.calculateScore(bitmap, detection, config)
                                    currentScore = score
                                    detectionStatus = "Person Detected (Conf: ${(detection.confidence*100).toInt()}%)"
                                } else {
                                    currentScore = 0f
                                    detectionStatus = "Searching..."
                                }
                            } catch (e: Exception) {
                                Log.e("DynamicDirector", "Analysis failed", e)
                            } finally {
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
                                imageAnalysis
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
            val appVersion = "v0.1.0"
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(16.dp),
                contentAlignment = Alignment.TopStart
            ) {
                Text(
                    text = "Dynamic Director $appVersion\nScore: ${currentScore.toInt()}%\n$detectionStatus",
                    color = if (currentScore > 85f) Color.Yellow else Color.Green,
                    style = MaterialTheme.typography.bodyLarge
                )
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
