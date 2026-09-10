package com.example.director

import android.graphics.Bitmap
import android.graphics.PointF
import android.util.Log
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.pose.Pose
import com.google.mlkit.vision.pose.PoseDetection
import com.google.mlkit.vision.pose.PoseLandmark
import com.google.mlkit.vision.pose.defaults.PoseDetectorOptions

/**
 * Result of pose analysis for a single frame.
 *
 * @param actionName The detected action name (null if no action detected)
 * @param confidence Confidence score (0.0 to 1.0)
 * @param rawScores Per-action confidence scores for UI display
 * @param landmarks Normalized landmark positions for overlay drawing
 */
data class PoseResult(
    val actionName: String?,
    val confidence: Float,
    val rawScores: Map<String, Float>,
    val landmarks: List<PointF>
)

/**
 * Pose analyzer that wraps ML Kit Pose Detection and feeds results through
 * the time-series buffer and action classifier pipeline.
 *
 * Architecture:
 *   ML Kit Pose Detection → PoseTimeSeriesBuffer → ActionClassifier → PoseResult
 *
 * The time-series buffer handles:
 * - FPS-invariant resampling (absorbs frame rate variations)
 * - Spatial normalization (shoulder-width based, body-size invariant)
 * - Dropped frame compensation
 *
 * The action classifier operates on normalized time-series data for
 * body-proportion-invariant action recognition.
 */
import android.content.Context

class PoseAnalyzer(context: Context) {
    // Stream mode for real-time pose detection
    private val options = PoseDetectorOptions.Builder()
        .setDetectorMode(PoseDetectorOptions.STREAM_MODE)
        .build()

    private val detector = PoseDetection.getClient(options)

    // Time-series buffer for FPS-invariant, normalized skeleton data
    private val timeSeriesBuffer = PoseTimeSeriesBuffer(
        windowSeconds = 1.0f,
        outputFrames = 30,
        minConfidence = 0.5f
    )

    // Improved action classifier operating on normalized time-series
    private val actionClassifier = ActionClassifier(context)

    fun analyze(
        bitmap: Bitmap,
        enabledActions: Set<String>,
        onResult: (PoseResult?) -> Unit,
        onComplete: () -> Unit
    ) {
        val image = InputImage.fromBitmap(bitmap, 0)
        detector.process(image)
            .addOnSuccessListener { pose ->
                if (pose.allPoseLandmarks.isEmpty()) {
                    timeSeriesBuffer.clear()
                    onResult(null)
                    return@addOnSuccessListener
                }

                // Feed raw pose data into the time-series buffer
                timeSeriesBuffer.addFrame(pose, bitmap.width, bitmap.height)

                // Extract landmarks for overlay drawing (raw normalized positions)
                val points = pose.allPoseLandmarks.map {
                    PointF(it.position.x / bitmap.width, it.position.y / bitmap.height)
                }

                // Classify action if buffer is ready
                val actionResult = if (timeSeriesBuffer.isReady()) {
                    val tensor = timeSeriesBuffer.getResampledTensor()
                    if (tensor != null) {
                        actionClassifier.classify(tensor, enabledActions)
                    } else {
                        null
                    }
                } else {
                    null
                }

                onResult(PoseResult(
                    actionName = actionResult?.actionName,
                    confidence = actionResult?.confidence ?: 0f,
                    rawScores = actionResult?.rawScores ?: emptyMap(),
                    landmarks = points
                ))
            }
            .addOnFailureListener { e ->
                Log.e("DynamicDirector", "Pose detection failed", e)
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
