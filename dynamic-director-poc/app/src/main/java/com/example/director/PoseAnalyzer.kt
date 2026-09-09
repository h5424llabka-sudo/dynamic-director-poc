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
        val nose = pose.getPoseLandmark(PoseLandmark.NOSE)

        if (lShoulder == null || rShoulder == null || lWrist == null || rWrist == null || lElbow == null || rElbow == null) {
            return null
        }
        
        val shoulderWidth = hypot(
            lShoulder.position.x - rShoulder.position.x,
            lShoulder.position.y - rShoulder.position.y
        )
        
        // Minimum confidence for all landmarks involved
        val minConfidence = 0.7f

        // 1. 拍手 (Clapping) - Both wrists very close together at chest level
        if (enabledActions.contains("Clapping")) {
            val wristDist = hypot(lWrist.position.x - rWrist.position.x, lWrist.position.y - rWrist.position.y)
            if (lWrist.inFrameLikelihood > minConfidence && rWrist.inFrameLikelihood > minConfidence) {
                // Wrists must be within 30% of shoulder width (very close)
                val isClose = wristDist < shoulderWidth * 0.3f
                // Both wrists should be between shoulder and hip height (chest area)
                val midShoulderY = (lShoulder.position.y + rShoulder.position.y) / 2f
                val lAtChest = lWrist.position.y > midShoulderY && lWrist.position.y < midShoulderY + shoulderWidth * 2f
                val rAtChest = rWrist.position.y > midShoulderY && rWrist.position.y < midShoulderY + shoulderWidth * 2f
                if (isClose && lAtChest && rAtChest) {
                    return "Clapping"
                }
            }
        }

        // 2. 万歳 (Banzai) - Both hands raised HIGH above head
        if (enabledActions.contains("Banzai")) {
            if (lWrist.inFrameLikelihood > minConfidence && rWrist.inFrameLikelihood > minConfidence
                && lElbow.inFrameLikelihood > minConfidence && rElbow.inFrameLikelihood > minConfidence) {
                // Reference: use nose Y if available, else shoulder Y - shoulderWidth as head proxy
                val headY = nose?.position?.y ?: (lShoulder.position.y - shoulderWidth)
                // Both wrists must be ABOVE head level
                val isLeftUp = lWrist.position.y < headY
                val isRightUp = rWrist.position.y < headY
                // Both elbows must also be above shoulder level
                val isLeftElbowUp = lElbow.position.y < lShoulder.position.y
                val isRightElbowUp = rElbow.position.y < rShoulder.position.y
                if (isLeftUp && isRightUp && isLeftElbowUp && isRightElbowUp) {
                    return "Banzai"
                }
            }
        }
        
        // 3. 指さし (Pointing) - One arm fully extended, the other NOT extended
        if (enabledActions.contains("Pointing")) {
            if (lWrist.inFrameLikelihood > minConfidence && rWrist.inFrameLikelihood > minConfidence) {
                val lArmLength = hypot(lShoulder.position.x - lWrist.position.x, lShoulder.position.y - lWrist.position.y)
                val rArmLength = hypot(rShoulder.position.x - rWrist.position.x, rShoulder.position.y - rWrist.position.y)
                // Arm must extend > 2x shoulder width for a clear point
                val extendThreshold = shoulderWidth * 2.0f
                // The OTHER arm should NOT be extended (asymmetry check)
                val retractThreshold = shoulderWidth * 1.2f
                // Wrist should be roughly at shoulder height (not raised high, not hanging low)
                val yTolerance = shoulderWidth * 0.8f
                
                val lExtended = lArmLength > extendThreshold && abs(lWrist.position.y - lShoulder.position.y) < yTolerance
                val rExtended = rArmLength > extendThreshold && abs(rWrist.position.y - rShoulder.position.y) < yTolerance
                val lRetracted = lArmLength < retractThreshold
                val rRetracted = rArmLength < retractThreshold
                
                if (lExtended && rRetracted) return "Pointing"
                if (rExtended && lRetracted) return "Pointing"
            }
        }

        // History-based Actions - require sufficient history
        if (poseHistory.size >= 8) {
            
            // 4. 手を振る (Waving) - Hand raised above head + clear left-right oscillation
            if (enabledActions.contains("Waving")) {
                val headY = nose?.position?.y ?: (lShoulder.position.y - shoulderWidth)
                // At least one wrist must be ABOVE head level (clearly raised to wave)
                val leftIsRaised = lWrist.position.y < headY && lWrist.inFrameLikelihood > minConfidence
                val rightIsRaised = rWrist.position.y < headY && rWrist.inFrameLikelihood > minConfidence
                
                if (leftIsRaised || rightIsRaised) {
                    val targetWrist = if (leftIsRaised) PoseLandmark.LEFT_WRIST else PoseLandmark.RIGHT_WRIST
                    
                    // Collect X positions from history
                    val xPositions = mutableListOf<Float>()
                    for (p in poseHistory) {
                        val w = p.getPoseLandmark(targetWrist) ?: continue
                        xPositions.add(w.position.x)
                    }
                    
                    if (xPositions.size >= 6) {
                        // Count direction reversals (sign changes in X delta)
                        var reversals = 0
                        var lastDelta = 0f
                        for (i in 1 until xPositions.size) {
                            val delta = xPositions[i] - xPositions[i - 1]
                            if (lastDelta != 0f && delta * lastDelta < 0f) {
                                // Direction changed
                                reversals++
                            }
                            if (abs(delta) > 5f) { // Ignore tiny jitter
                                lastDelta = delta
                            }
                        }
                        // Must have at least 2 direction reversals (= back-and-forth motion)
                        // AND total X range must be large enough
                        val minX = xPositions.min()
                        val maxX = xPositions.max()
                        if (reversals >= 2 && (maxX - minX) > shoulderWidth * 0.8f) {
                            return "Waving"
                        }
                    }
                }
            }
            
            // 5. 投げる (Throwing) - Rapid downward arm sweep
            if (enabledActions.contains("Throwing")) {
                // Check both arms for throwing motion
                for (wristLandmark in listOf(PoseLandmark.LEFT_WRIST, PoseLandmark.RIGHT_WRIST)) {
                    val shoulderLandmark = if (wristLandmark == PoseLandmark.LEFT_WRIST) PoseLandmark.LEFT_SHOULDER else PoseLandmark.RIGHT_SHOULDER
                    
                    val oldPose = poseHistory.first()
                    val oldWrist = oldPose.getPoseLandmark(wristLandmark)
                    val oldShoulder = oldPose.getPoseLandmark(shoulderLandmark)
                    val curWrist = pose.getPoseLandmark(wristLandmark)
                    
                    if (oldWrist != null && oldShoulder != null && curWrist != null
                        && curWrist.inFrameLikelihood > minConfidence) {
                        // Old wrist must have been ABOVE shoulder (wind-up position)
                        val wasWindedUp = oldWrist.position.y < oldShoulder.position.y
                        // Current wrist must be BELOW shoulder (follow-through)
                        val isFollowThrough = curWrist.position.y > (lShoulder.position.y + rShoulder.position.y) / 2f
                        // Total Y displacement must be significant
                        val yDiff = curWrist.position.y - oldWrist.position.y
                        if (wasWindedUp && isFollowThrough && yDiff > shoulderWidth * 1.5f) {
                            return "Throwing"
                        }
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
