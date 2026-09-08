package com.example.director

import android.graphics.Bitmap
import android.util.Log
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions

data class FaceResult(
    val smilingProbability: Float,
    val headEulerAngleY: Float, // Left/Right (Yaw)
    val headEulerAngleZ: Float, // Tilt (Roll)
    val headEulerAngleX: Float, // Up/Down (Pitch)
    val x1: Float,
    val y1: Float,
    val x2: Float,
    val y2: Float
)

class FaceAnalyzer {
    // High-accuracy mode with all classifications (smile, eyes open)
    private val options = FaceDetectorOptions.Builder()
        .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
        .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL)
        .build()

    private val detector = FaceDetection.getClient(options)

    fun analyze(bitmap: Bitmap, onResult: (FaceResult?) -> Unit, onComplete: () -> Unit) {
        val image = InputImage.fromBitmap(bitmap, 0)
        detector.process(image)
            .addOnSuccessListener { faces ->
                if (faces.isNotEmpty()) {
                    // Find the most prominent face (largest bounding box)
                    val bestFace = faces.maxByOrNull { it.boundingBox.width() * it.boundingBox.height() }
                    
                    if (bestFace != null) {
                        val smileProb = bestFace.smilingProbability ?: 0f
                        val eulerY = bestFace.headEulerAngleY
                        val eulerZ = bestFace.headEulerAngleZ
                        val eulerX = bestFace.headEulerAngleX
                        
                        val w = bitmap.width.toFloat()
                        val h = bitmap.height.toFloat()
                        val x1 = bestFace.boundingBox.left / w
                        val y1 = bestFace.boundingBox.top / h
                        val x2 = bestFace.boundingBox.right / w
                        val y2 = bestFace.boundingBox.bottom / h
                        
                        onResult(FaceResult(smileProb, eulerY, eulerZ, eulerX, x1, y1, x2, y2))
                        return@addOnSuccessListener
                    }
                }
                onResult(null)
            }
            .addOnFailureListener { e ->
                Log.e("DynamicDirector", "Face detection failed", e)
                onResult(null)
            }
            .addOnCompleteListener {
                onComplete()
            }
    }

    fun close() {
        detector.close()
    }
}
