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
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.verticalScroll
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
        
        var detectionStatus by remember { mutableStateOf("Waiting for AI...") }
        var currentFace by remember { mutableStateOf<FaceResult?>(null) }
        var currentPose by remember { mutableStateOf<PoseResult?>(null) }
        var triggerStatus by remember { mutableStateOf("") }
        var actionScoresText by remember { mutableStateOf("") }
        var imageSize by remember { mutableStateOf(androidx.compose.ui.geometry.Size.Zero) }
        
        // Initialize AI models
        val faceAnalyzer = remember { FaceAnalyzer() }
        val poseAnalyzer = remember { PoseAnalyzer() }
        val config = remember { ConfigManager.loadConfig(context) }
        
        // Initialize TriggerController
        val triggerController = remember { mutableStateOf(TriggerController()) }
        
        // Settings State
        var isSettingsOpen by remember { mutableStateOf(false) }
        val smileThresholdState = remember { mutableStateOf(80f) }
        val cooldownSecondsState = remember { mutableStateOf(3f) }
        val confirmationFramesState = remember { mutableStateOf(3f) }
        
        // Action states (Enabled, Threshold)
        val banzaiState = remember { mutableStateOf(true) }
        val banzaiThreshold = remember { mutableStateOf(70f) }
        
        val pointingState = remember { mutableStateOf(true) }
        val pointingThreshold = remember { mutableStateOf(70f) }
        
        val wavingState = remember { mutableStateOf(true) }
        val wavingThreshold = remember { mutableStateOf(70f) }
        
        val throwingState = remember { mutableStateOf(true) }
        val throwingThreshold = remember { mutableStateOf(80f) } // Stricter for throwing
        
        val clappingState = remember { mutableStateOf(true) }
        val clappingThreshold = remember { mutableStateOf(20f) } // More sensitive for clapping
        
        val getEnabledActions = {
            val set = mutableSetOf<String>()
            if (banzaiState.value) set.add("Banzai")
            if (pointingState.value) set.add("Pointing")
            if (wavingState.value) set.add("Waving")
            if (throwingState.value) set.add("Throwing")
            if (clappingState.value) set.add("Clapping")
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
                            .setTargetResolution(android.util.Size(1920, 1080))
                            .build()
                            
                        val imageCapture = ImageCapture.Builder()
                            .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                            .build()
                            
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
                                
                                imageSize = androidx.compose.ui.geometry.Size(bitmap.width.toFloat(), bitmap.height.toFloat())
                                
                                // Chain Analyzers: Face -> Pose -> TriggerController
                                faceAnalyzer.analyze(bitmap, { faceResult ->
                                    currentFace = faceResult
                                    
                                    val enabledActions = getEnabledActions()
                                    poseAnalyzer.analyze(bitmap, enabledActions, { poseResult ->
                                        currentPose = poseResult
                                        
                                        // Build action scores text for dashboard
                                        val scoresBuilder = StringBuilder()
                                        poseResult?.rawScores?.forEach { (name, score) ->
                                            scoresBuilder.append("  $name: ${(score * 100).toInt()}%\n")
                                        }
                                        actionScoresText = scoresBuilder.toString()
                                        
                                        // Convert to ActionResult for TriggerController
                                        val actionResult = poseResult?.let {
                                            ActionResult(
                                                actionName = it.actionName,
                                                confidence = it.confidence,
                                                rawScores = it.rawScores
                                            )
                                        }
                                        
                                        val faceScore = faceResult?.let { ScoringEngine.calculateFaceScore(it) } ?: 0f
                                        val smileProbability = faceResult?.smilingProbability ?: 0f
                                        
                                        // TriggerController decides whether to fire
                                        val triggerEvent = triggerController.value.update(
                                            actionResult = actionResult,
                                            faceScore = faceScore,
                                            smileProbability = smileProbability
                                        )
                                        
                                        when (triggerEvent) {
                                            is TriggerEvent.Fire -> {
                                                if (!isTakingPhoto) {
                                                    isTakingPhoto = true
                                                    detectionStatus = "📸 ${triggerEvent.reason}"
                                                    triggerStatus = "🎯 FIRED! (${(triggerEvent.confidence * 100).toInt()}%)"
                                                    
                                                    // Zero-lag capture: Save the exact bitmap we just analyzed
                                                    Executors.newSingleThreadExecutor().execute {
                                                        try {
                                                            val now = System.currentTimeMillis()
                                                            val mutableBitmap = bitmap.copy(android.graphics.Bitmap.Config.ARGB_8888, true)
                                                            val canvas = android.graphics.Canvas(mutableBitmap)
                                                            val paint = android.graphics.Paint().apply {
                                                                color = android.graphics.Color.YELLOW
                                                                textSize = 48f
                                                                style = android.graphics.Paint.Style.FILL
                                                                isAntiAlias = true
                                                            }
                                                            val bgPaint = android.graphics.Paint().apply {
                                                                color = android.graphics.Color.argb(128, 0, 0, 0)
                                                                style = android.graphics.Paint.Style.FILL
                                                            }
                                                            val triggerText = "Trigger: ${triggerEvent.reason}"
                                                            val confText = "Confidence: ${(triggerEvent.confidence * 100).toInt()}%"
                                                            val textWidth = maxOf(paint.measureText(triggerText), paint.measureText(confText))
                                                            canvas.drawRect(20f, 20f, 60f + textWidth, 140f, bgPaint)
                                                            canvas.drawText(triggerText, 40f, 70f, paint)
                                                            canvas.drawText(confText, 40f, 120f, paint)
                                                            
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
                                                                Log.d("DynamicDirector", "Zero-lag photo saved: $uri | Trigger: ${triggerEvent.reason}")
                                                            }
                                                        } catch (e: Exception) {
                                                            Log.e("DynamicDirector", "Failed to save zero-lag image", e)
                                                        } finally {
                                                            isTakingPhoto = false
                                                        }
                                                    }
                                                }
                                            }
                                            is TriggerEvent.Cooldown -> {
                                                detectionStatus = "❄️ Cooldown (${triggerEvent.remainingMs / 1000}s)"
                                                triggerStatus = ""
                                            }
                                            is TriggerEvent.Idle -> {
                                                detectionStatus = triggerEvent.status
                                                triggerStatus = ""
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
            val appVersion = "v0.5.0"
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
                
                val poseText = "Pose: ${currentPose?.actionName ?: "None"}" +
                    if (currentPose?.confidence ?: 0f > 0f) " (${(currentPose!!.confidence * 100).toInt()}%)" else ""
                
                val dashboardText = buildString {
                    appendLine("Dynamic Director $appVersion")
                    appendLine("Status: $detectionStatus")
                    appendLine(poseText)
                    appendLine(faceText)
                    if (actionScoresText.isNotEmpty()) {
                        appendLine("--- Action Scores ---")
                        append(actionScoresText)
                    }
                    if (triggerStatus.isNotEmpty()) {
                        appendLine(triggerStatus)
                    }
                }
                
                Text(
                    text = dashboardText,
                    color = when {
                        triggerStatus.isNotEmpty() -> Color.Yellow
                        currentPose?.actionName != null -> Color(0xFF00E5FF)  // Cyan
                        else -> Color.Green
                    },
                    style = MaterialTheme.typography.bodyLarge
                )
            }
            
            // Helper function for FILL_CENTER coordinate mapping
            fun mapCoordinate(x: Float, y: Float, viewWidth: Float, viewHeight: Float, imgWidth: Float, imgHeight: Float): androidx.compose.ui.geometry.Offset {
                if (imgWidth == 0f || imgHeight == 0f) return androidx.compose.ui.geometry.Offset(x * viewWidth, y * viewHeight)
                
                val scale = maxOf(viewWidth / imgWidth, viewHeight / imgHeight)
                val offsetX = (viewWidth - imgWidth * scale) / 2f
                val offsetY = (viewHeight - imgHeight * scale) / 2f
                
                return androidx.compose.ui.geometry.Offset(
                    x = (x * imgWidth) * scale + offsetX,
                    y = (y * imgHeight) * scale + offsetY
                )
            }

            // Draw BBox Overlay
            currentFace?.let { face ->
                androidx.compose.foundation.Canvas(modifier = Modifier.fillMaxSize()) {
                    val w = size.width
                    val h = size.height
                    
                    val topLeft = mapCoordinate(face.x1, face.y1, w, h, imageSize.width, imageSize.height)
                    val bottomRight = mapCoordinate(face.x2, face.y2, w, h, imageSize.width, imageSize.height)
                    
                    drawRect(
                        color = Color.Green,
                        topLeft = topLeft,
                        size = androidx.compose.ui.geometry.Size(bottomRight.x - topLeft.x, bottomRight.y - topLeft.y),
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
                            center = mapCoordinate(point.x, point.y, w, h, imageSize.width, imageSize.height)
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
                            // --- Trigger Settings ---
                            Text("Trigger Settings", style = MaterialTheme.typography.titleSmall)
                            androidx.compose.foundation.layout.Spacer(modifier = Modifier.height(8.dp))
                            
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
                            
                            androidx.compose.foundation.layout.Spacer(modifier = Modifier.height(8.dp))
                            Text("Confirmation Frames: ${confirmationFramesState.value.toInt()}")
                            androidx.compose.material3.Slider(
                                value = confirmationFramesState.value,
                                onValueChange = { confirmationFramesState.value = it },
                                valueRange = 1f..10f,
                                steps = 8
                            )
                            
                            // Apply button for trigger settings
                            androidx.compose.material3.TextButton(
                                onClick = {
                                    val newThresholds = mapOf(
                                        "Banzai" to banzaiThreshold.value / 100f,
                                        "Pointing" to pointingThreshold.value / 100f,
                                        "Waving" to wavingThreshold.value / 100f,
                                        "Throwing" to throwingThreshold.value / 100f,
                                        "Clapping" to clappingThreshold.value / 100f
                                    )
                                    triggerController.value = triggerController.value.updateConfig(
                                        newActivationThresholds = newThresholds,
                                        newCooldownMs = (cooldownSecondsState.value * 1000).toLong(),
                                        newConfirmationFrames = confirmationFramesState.value.toInt(),
                                        newSmileThreshold = smileThresholdState.value / 100f
                                    )
                                }
                            ) {
                                Text("Apply Trigger Settings")
                            }
                            
                            androidx.compose.foundation.layout.Spacer(modifier = Modifier.height(16.dp))
                            
                            // --- Action Toggles & Thresholds ---
                            Text("Action Settings:", style = MaterialTheme.typography.titleSmall)
                            data class ActionSetting(
                                val name: String, 
                                val enabled: androidx.compose.runtime.MutableState<Boolean>, 
                                val threshold: androidx.compose.runtime.MutableState<Float>
                            )
                            val actionSettings = listOf(
                                ActionSetting("Banzai", banzaiState, banzaiThreshold),
                                ActionSetting("Pointing", pointingState, pointingThreshold),
                                ActionSetting("Waving", wavingState, wavingThreshold),
                                ActionSetting("Throwing", throwingState, throwingThreshold),
                                ActionSetting("Clapping", clappingState, clappingThreshold)
                            )
                            
                            actionSettings.forEach { setting ->
                                androidx.compose.foundation.layout.Row(
                                    verticalAlignment = Alignment.CenterVertically,
                                    modifier = Modifier.fillMaxWidth()
                                ) {
                                    Text(
                                        "${setting.name} (${setting.threshold.value.toInt()}%)", 
                                        modifier = Modifier.weight(1f)
                                    )
                                    androidx.compose.material3.Switch(
                                        checked = setting.enabled.value,
                                        onCheckedChange = { setting.enabled.value = it }
                                    )
                                }
                                if (setting.enabled.value) {
                                    androidx.compose.material3.Slider(
                                        value = setting.threshold.value,
                                        onValueChange = { setting.threshold.value = it },
                                        valueRange = 0f..100f
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
