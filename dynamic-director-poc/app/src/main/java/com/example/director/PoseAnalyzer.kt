package com.example.director

import android.graphics.Bitmap
import android.graphics.PointF
import android.util.Log
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.pose.Pose
import com.google.mlkit.vision.pose.PoseDetection
import com.google.mlkit.vision.pose.PoseLandmark
import com.google.mlkit.vision.pose.defaults.PoseDetectorOptions
import kotlin.math.abs
import kotlin.math.hypot

data class PoseResult(
    val actionName: String?,
    val score: Float, // Confidence or magnitude of the action (0 to 100)
    val landmarks: List<PointF>
)

class PoseAnalyzer {
    // Fast mode is sufficient for basic gesture detection
    private val options = PoseDetectorOptions.Builder()
        .setDetectorMode(PoseDetectorOptions.STREAM_MODE)
        .build()

    private val detector = PoseDetection.getClient(options)
    
    // History buffer for motion-based actions (waving, throwing)
    private val historyMax = 10
    private val poseHistory = mutableListOf<Pose>()

    fun analyze(bitmap: Bitmap, enabledActions: Set<String>, onResult: (PoseResult?) -> Unit, onComplete: () -> Unit) {
        val image = InputImage.fromBitmap(bitmap, 0)
        detector.process(image)
            .addOnSuccessListener { pose ->
                if (pose.allPoseLandmarks.isEmpty()) {
                    poseHistory.clear()
                    onResult(null)
                    return@addOnSuccessListener
                }
                
                // Add to history
                poseHistory.add(pose)
                if (poseHistory.size > historyMax) {
                    poseHistory.removeAt(0)
                }

                // Evaluate Actions
                val action = evaluateActions(pose, enabledActions)
                
                // Extract points for drawing
                val points = pose.allPoseLandmarks.map { PointF(it.position.x / bitmap.width, it.position.y / bitmap.height) }
                
                onResult(PoseResult(action, if (action != null) 100f else 0f, points))
            }
            .addOnFailureListener { e ->
                Log.e("DynamicDirector", "Pose detection failed", e)
                onResult(null)
            }
            .addOnCompleteListener {
                onComplete()
            }
    }

    private fun evaluateActions(pose: Pose, enabledActions: Set<String>): String? {
        val lShoulder = pose.getPoseLandmark(PoseLandmark.LEFT_SHOULDER)
        val rShoulder = pose.getPoseLandmark(PoseLandmark.RIGHT_SHOULDER)
        val lElbow = pose.getPoseLandmark(PoseLandmark.LEFT_ELBOW)
        val rElbow = pose.getPoseLandmark(PoseLandmark.RIGHT_ELBOW)
        val lWrist = pose.getPoseLandmark(PoseLandmark.LEFT_WRIST)
        val rWrist = pose.getPoseLandmark(PoseLandmark.RIGHT_WRIST)

        if (lShoulder == null || rShoulder == null || lWrist == null || rWrist == null || lElbow == null || rElbow == null) {
            return null
        }
        
        // 1. 拍手 (Clapping)
        if (enabledActions.contains("Clapping")) {
            val wristDist = hypot(lWrist.position.x - rWrist.position.x, lWrist.position.y - rWrist.position.y)
            val shoulderWidth = hypot(lShoulder.position.x - rShoulder.position.x, lShoulder.position.y - rShoulder.position.y)
            // If wrists are very close to each other (less than half shoulder width) and below nose
            if (wristDist < shoulderWidth * 0.5f && lWrist.inFrameLikelihood > 0.5f && rWrist.inFrameLikelihood > 0.5f) {
                // Also check they are roughly at chest height
                if (lWrist.position.y > lShoulder.position.y - 50) {
                    return "Clapping"
                }
            }
        }

        // 2. 万歳 (Banzai)
        if (enabledActions.contains("Banzai")) {
            val shoulderWidth = hypot(lShoulder.position.x - rShoulder.position.x, lShoulder.position.y - rShoulder.position.y)
            // Y goes down. So wrist.y < shoulder.y means wrist is HIGHER than shoulder.
            // We use a margin to ensure it's significantly higher.
            val margin = shoulderWidth * 0.5f
            val isLeftBanzai = lWrist.position.y < (lShoulder.position.y - margin)
            val isRightBanzai = rWrist.position.y < (rShoulder.position.y - margin)
            if (isLeftBanzai && isRightBanzai && lWrist.inFrameLikelihood > 0.5f && rWrist.inFrameLikelihood > 0.5f) {
                return "Banzai"
            }
        }
        
        // 3. 指さし (Pointing)
        if (enabledActions.contains("Pointing")) {
            val shoulderWidth = hypot(lShoulder.position.x - rShoulder.position.x, lShoulder.position.y - rShoulder.position.y)
            val margin = shoulderWidth * 0.5f
            // Elbow is almost straight, and wrist is far from shoulder
            val lArmLength = hypot(lShoulder.position.x - lWrist.position.x, lShoulder.position.y - lWrist.position.y)
            val rArmLength = hypot(rShoulder.position.x - rWrist.position.x, rShoulder.position.y - rWrist.position.y)
            // If arm length is > 1.5x shoulder width, it's extended
            if (lArmLength > shoulderWidth * 1.5f && lWrist.position.y > lShoulder.position.y - margin) {
                return "Pointing"
            }
            if (rArmLength > shoulderWidth * 1.5f && rWrist.position.y > rShoulder.position.y - margin) {
                return "Pointing"
            }
        }

        // History-based Actions
        if (poseHistory.size >= 5) {
            val shoulderWidth = hypot(lShoulder.position.x - rShoulder.position.x, lShoulder.position.y - rShoulder.position.y)
            
            // 4. 手を振る (Waving)
            if (enabledActions.contains("Waving")) {
                // Look for wrist X oscillation while wrist is raised
                val leftIsRaised = lWrist.position.y < lShoulder.position.y
                val rightIsRaised = rWrist.position.y < rShoulder.position.y
                
                if (leftIsRaised || rightIsRaised) {
                    val targetWrist = if (leftIsRaised) PoseLandmark.LEFT_WRIST else PoseLandmark.RIGHT_WRIST
                    var minX = Float.MAX_VALUE
                    var maxX = Float.MIN_VALUE
                    for (p in poseHistory) {
                        val w = p.getPoseLandmark(targetWrist) ?: continue
                        if (w.position.x < minX) minX = w.position.x
                        if (w.position.x > maxX) maxX = w.position.x
                    }
                    // If X variation is large (e.g. > shoulderWidth * 0.5)
                    if (maxX - minX > shoulderWidth * 0.6f) {
                        return "Waving"
                    }
                }
            }
            
            // 5. 投げる (Throwing)
            if (enabledActions.contains("Throwing")) {
                // Rapid downward motion of wrist
                val oldestPose = poseHistory.first()
                val oldRWrist = oldestPose.getPoseLandmark(PoseLandmark.RIGHT_WRIST)
                if (oldRWrist != null) {
                    val yDiff = rWrist.position.y - oldRWrist.position.y // Positive means moving DOWN
                    if (yDiff > shoulderWidth * 1.0f) { // Moved down by at least shoulder width quickly
                        return "Throwing"
                    }
                }
            }
        }

        return null
    }

    fun close() {
        detector.close()
    }
}
