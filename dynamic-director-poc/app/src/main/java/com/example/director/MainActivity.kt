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
        
        // Settings State
        var isSettingsOpen by remember { mutableStateOf(false) }
        val smileThresholdState = remember { mutableStateOf(80f) }
        val cooldownSecondsState = remember { mutableStateOf(3f) }
        val enableBanzaiState = remember { mutableStateOf(true) }
        val enablePointingState = remember { mutableStateOf(true) }
        val enableWavingState = remember { mutableStateOf(true) }
        val enableThrowingState = remember { mutableStateOf(true) }
        val enableClappingState = remember { mutableStateOf(true) }
        
        val getEnabledActions = {
            val set = mutableSetOf<String>()
            if (enableBanzaiState.value) set.add("Banzai")
            if (enablePointingState.value) set.add("Pointing")
            if (enableWavingState.value) set.add("Waving")
            if (enableThrowingState.value) set.add("Throwing")
            if (enableClappingState.value) set.add("Clapping")
            set
        }

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
                                    
                                    val enabledActions = getEnabledActions()
                                    poseAnalyzer.analyze(bitmap, enabledActions, { poseResult ->
                                        currentPose = poseResult
                                        
                                        val faceScore = faceResult?.let { ScoringEngine.calculateFaceScore(it) } ?: 0f
                                        val poseScore = poseResult?.score ?: 0f
                                        val score = maxOf(faceScore, poseScore)
                                        currentScore = score
                                        
                                        // Phase 6 & 7: Auto Shutter Logic
                                        val now = System.currentTimeMillis()
                                        val cooldownMs = (cooldownSecondsState.value * 1000).toLong()
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
                                            
                                            // Shutter threshold (Face > smileThreshold or Pose = 100)
                                            val isSmileTrigger = faceResult != null && (faceResult.smilingProbability * 100) >= smileThresholdState.value
                                            val isPoseTrigger = poseResult != null && poseResult.actionName != null
                                            if (isSmileTrigger || isPoseTrigger) {
                                                consecutiveHighScores++
                                                if (consecutiveHighScores >= 3) {
                                                    isTakingPhoto = true
                                                    lastCaptureTime = now
                                                    consecutiveHighScores = 0
                                                    
                                                    val triggerReason = if (poseResult != null && poseResult.actionName != null) {
                                                        "Action: ${poseResult.actionName}"
                                                    } else if (faceResult != null) {
                                                        "Smile: ${(faceResult.smilingProbability * 100).toInt()}%"
                                                    } else {
                                                        "Unknown"
                                                    }
                                                    
                                                    imageCapture.takePicture(
                                                        ContextCompat.getMainExecutor(ctx),
                                                        object : ImageCapture.OnImageCapturedCallback() {
                                                            override fun onCaptureSuccess(image: androidx.camera.core.ImageProxy) {
                                                                try {
                                                                    val rawBitmap = image.toBitmap()
                                                                    val matrix = android.graphics.Matrix()
                                                                    matrix.postRotate(image.imageInfo.rotationDegrees.toFloat())
                                                                    val bitmap = android.graphics.Bitmap.createBitmap(rawBitmap, 0, 0, rawBitmap.width, rawBitmap.height, matrix, true)
                                                                    
                                                                    val mutableBitmap = bitmap.copy(android.graphics.Bitmap.Config.ARGB_8888, true)
                                                                    val canvas = android.graphics.Canvas(mutableBitmap)
                                                                    val paint = android.graphics.Paint().apply {
                                                                        color = android.graphics.Color.YELLOW
                                                                        textSize = 120f
                                                                        style = android.graphics.Paint.Style.FILL
                                                                        isAntiAlias = true
                                                                    }
                                                                    val bgPaint = android.graphics.Paint().apply {
                                                                        color = android.graphics.Color.argb(128, 0, 0, 0)
                                                                        style = android.graphics.Paint.Style.FILL
                                                                    }
                                                                    val triggerText = "Trigger: $triggerReason"
                                                                    val textWidth = paint.measureText(triggerText)
                                                                    canvas.drawRect(40f, 40f, 80f + textWidth, 200f, bgPaint)
                                                                    canvas.drawText(triggerText, 60f, 150f, paint)
                                                                    
                                                                    val contentValues = android.content.ContentValues().apply {
                                                                        put(android.provider.MediaStore.MediaColumns.DISPLAY_NAME, "director_capture_${now}.jpg")
                                                                        put(android.provider.MediaStore.MediaColumns.MIME_TYPE, "image/jpeg")
                                                                        if (android.os.Build.VERSION.SDK_INT > android.os.Build.VERSION_CODES.P) {
                                                                            put(android.provider.MediaStore.Images.Media.RELATIVE_PATH, "Pictures/DynamicDirector")
                                                                        }
                                                                    }
                                                                    val uri = ctx.contentResolver.insert(android.provider.MediaStore.Images.Media.EXTERNAL_CONTENT_URI, contentValues)
                                                                    if (uri != null) {
                                                                        ctx.contentResolver.openOutputStream(uri)?.use { out ->
                                                                            mutableBitmap.compress(android.graphics.Bitmap.CompressFormat.JPEG, 95, out)
                                                                        }
                                                                        Log.d("DynamicDirector", "Photo saved to Gallery with text: $uri")
                                                                    }
                                                                } catch (e: Exception) {
                                                                    Log.e("DynamicDirector", "Failed to process image", e)
                                                                } finally {
                                                                    image.close()
                                                                    isTakingPhoto = false
                                                                }
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
            val appVersion = "v0.4.0"
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
            
            // Settings Button
            androidx.compose.material3.FloatingActionButton(
                onClick = { isSettingsOpen = true },
                modifier = Modifier
                    .align(Alignment.TopEnd)
                    .padding(16.dp)
            ) {
                Text("⚙️")
            }
            
            // Settings Dialog
            if (isSettingsOpen) {
                androidx.compose.material3.AlertDialog(
                    onDismissRequest = { isSettingsOpen = false },
                    title = { Text("Settings") },
                    text = {
                        androidx.compose.foundation.layout.Column(
                            modifier = Modifier
                                .fillMaxWidth()
                                .verticalScroll(androidx.compose.foundation.rememberScrollState())
                        ) {
                            Text("Smile Threshold: ${smileThresholdState.value.toInt()}%")
                            androidx.compose.material3.Slider(
                                value = smileThresholdState.value,
                                onValueChange = { smileThresholdState.value = it },
                                valueRange = 0f..100f
                            )
                            
                            androidx.compose.foundation.layout.Spacer(modifier = Modifier.height(8.dp))
                            Text("Cooldown: ${cooldownSecondsState.value.toInt()}s")
                            androidx.compose.material3.Slider(
                                value = cooldownSecondsState.value,
                                onValueChange = { cooldownSecondsState.value = it },
                                valueRange = 1f..10f
                            )
                            
                            androidx.compose.foundation.layout.Spacer(modifier = Modifier.height(16.dp))
                            Text("Enabled Actions:")
                            val actionToggles = listOf(
                                "Banzai" to enableBanzaiState,
                                "Pointing" to enablePointingState,
                                "Waving" to enableWavingState,
                                "Throwing" to enableThrowingState,
                                "Clapping" to enableClappingState
                            )
                            actionToggles.forEach { (name, state) ->
                                androidx.compose.foundation.layout.Row(
                                    verticalAlignment = Alignment.CenterVertically,
                                    modifier = Modifier.fillMaxWidth()
                                ) {
                                    Text(name, modifier = Modifier.weight(1f))
                                    androidx.compose.material3.Switch(
                                        checked = state.value,
                                        onCheckedChange = { state.value = it }
                                    )
                                }
                            }
                        }
                    },
                    confirmButton = {
                        androidx.compose.material3.TextButton(onClick = { isSettingsOpen = false }) {
                            Text("Close")
                        }
                    }
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
