package com.example.director

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.lite.task.core.BaseOptions
import org.tensorflow.lite.task.vision.detector.ObjectDetector
import org.tensorflow.lite.support.image.TensorImage

data class DetectionResult(
    val x1: Float,
    val y1: Float,
    val x2: Float,
    val y2: Float,
    val confidence: Float,
    val classId: Int
)

class YoloDetector(context: Context, modelName: String = "efficientdet_lite0.tflite") {
    private var objectDetector: ObjectDetector? = null

    init {
        try {
            val options = ObjectDetector.ObjectDetectorOptions.builder()
                .setMaxResults(5)
                .setScoreThreshold(0.5f)
                .setBaseOptions(BaseOptions.builder().setNumThreads(4).build())
                .build()
            
            objectDetector = ObjectDetector.createFromFileAndOptions(context, modelName, options)
            Log.d("DynamicDirector", "ObjectDetector initialized successfully")
        } catch (e: Exception) {
            Log.e("DynamicDirector", "Failed to initialize ObjectDetector", e)
        }
    }

    val isMockMode: Boolean
        get() = objectDetector == null

    private var mockOffset = 0f

    fun detect(bitmap: Bitmap): DetectionResult? {
        val detector = objectDetector
        if (detector == null) {
            // Mock Mode: return a moving fake BBox in the center of the screen
            mockOffset += 0.01f
            if (mockOffset > 0.1f) mockOffset = -0.1f
            val cx = 0.5f + mockOffset
            val cy = 0.5f
            val w = 0.3f
            val h = 0.5f
            return DetectionResult(
                x1 = cx - w/2,
                y1 = cy - h/2,
                x2 = cx + w/2,
                y2 = cy + h/2,
                confidence = 0.99f,
                classId = 0
            )
        }

        try {
            val image = TensorImage.fromBitmap(bitmap)
            val results = detector.detect(image)
            
            // Find best "person" detection
            var bestPerson: DetectionResult? = null
            var bestScore = 0f
            
            val imageWidth = bitmap.width.toFloat()
            val imageHeight = bitmap.height.toFloat()

            for (result in results) {
                val category = result.categories.firstOrNull() ?: continue
                if (category.label.equals("person", ignoreCase = true)) {
                    if (category.score > bestScore) {
                        bestScore = category.score
                        val bbox = result.boundingBox
                        
                        // Convert to relative coordinates (0.0 - 1.0) for ScoringEngine
                        val x1 = bbox.left / imageWidth
                        val y1 = bbox.top / imageHeight
                        val x2 = bbox.right / imageWidth
                        val y2 = bbox.bottom / imageHeight
                        
                        bestPerson = DetectionResult(x1, y1, x2, y2, category.score, 0)
                    }
                }
            }
            return bestPerson
        } catch (e: Exception) {
            e.printStackTrace()
            return null
        }
    }

    fun close() {
        objectDetector?.close()
        objectDetector = null
    }
}
