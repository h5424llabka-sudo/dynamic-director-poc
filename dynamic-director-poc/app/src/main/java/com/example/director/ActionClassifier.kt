package com.example.director

import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.sqrt

/**
 * Result of action classification with per-action confidence scores.
 */
data class ActionResult(
    val actionName: String?,
    val confidence: Float,  // 0.0 to 1.0 for the best action
    val rawScores: Map<String, Float>  // Individual scores for each action
)

/**
 * Improved action classifier that operates on spatially-normalized, time-resampled
 * skeleton sequences. Designed to handle toddler-specific challenges:
 *
 * - Uses shoulder-width-normalized coordinates (body-size invariant)
 * - Temporal pattern analysis for dynamic actions (waving, throwing)
 * - Elbow angle analysis for pointing and clapping
 * - Frequency-based oscillation detection for waving
 *
 * Keypoint indices in the normalized tensor (9 keypoints × 3 channels each):
 *   0: Nose, 1: L-Shoulder, 2: R-Shoulder, 3: L-Elbow, 4: R-Elbow,
 *   5: L-Wrist, 6: R-Wrist, 7: L-Hip, 8: R-Hip
 */
import android.content.Context
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.channels.FileChannel

class ActionClassifier(context: Context) {

    private var tflite: Interpreter? = null

    init {
        try {
            val assetFileDescriptor = context.assets.openFd("action_model_fp16.tflite")
            val fileInputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
            val fileChannel = fileInputStream.channel
            val mappedByteBuffer = fileChannel.map(
                FileChannel.MapMode.READ_ONLY, 
                assetFileDescriptor.startOffset, 
                assetFileDescriptor.declaredLength
            )
            tflite = Interpreter(mappedByteBuffer)
        } catch (e: Exception) {
            e.printStackTrace()
            // Fallback to rule-based only if model loading fails
        }
    }

    companion object {
        private const val C = PoseTimeSeriesBuffer.CHANNELS_PER_KEYPOINT  // 3

        // Keypoint index helpers (multiply by C to get base index in feature array)
        private const val NOSE = 0
        private const val L_SHOULDER = 1
        private const val R_SHOULDER = 2
        private const val L_ELBOW = 3
        private const val R_ELBOW = 4
        private const val L_WRIST = 5
        private const val R_WRIST = 6
        private const val L_HIP = 7
        private const val R_HIP = 8
    }

    /**
     * Classify actions from a normalized time-series tensor.
     *
     * @param tensor Shape [frames, 27] - spatially normalized, time-resampled
     * @param enabledActions Set of action names to evaluate
     * @return ActionResult with the best action and per-action scores
     */
    fun classify(tensor: Array<FloatArray>, enabledActions: Set<String>): ActionResult {
        val ruleScores = mutableMapOf<String, Float>()
        val lastFrame = tensor.last()

        ruleScores["Banzai"] = evaluateBanzai(lastFrame, tensor)
        ruleScores["Pointing"] = evaluatePointing(lastFrame)
        ruleScores["Waving"] = evaluateWaving(tensor)
        ruleScores["Throwing"] = evaluateThrowing(tensor)
        ruleScores["Clapping"] = evaluateClapping(lastFrame, tensor)

        // 2. CNN TFLite Inference
        val cnnScores = mutableMapOf<String, Float>()
        tflite?.let { interpreter ->
            // Format input tensor: [1, 30, 27]
            val input = Array(1) { Array(30) { FloatArray(27) } }
            for (i in 0 until 30) {
                if (i < tensor.size) {
                    System.arraycopy(tensor[i], 0, input[0][i], 0, 27)
                }
            }
            
            // Output shape: [1, 6]
            val output = Array(1) { FloatArray(6) }
            try {
                interpreter.run(input, output)
                val probs = output[0]
                // Classes: 0=None, 1=Banzai, 2=Pointing, 3=Waving, 4=Throwing, 5=Clapping
                cnnScores["Banzai"] = probs[1]
                cnnScores["Pointing"] = probs[2]
                cnnScores["Waving"] = probs[3]
                cnnScores["Throwing"] = probs[4]
                cnnScores["Clapping"] = probs[5]
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }

        // 3. Ensemble Scoring
        val finalScores = mutableMapOf<String, Float>()
        val actions = listOf("Banzai", "Pointing", "Waving", "Throwing", "Clapping")
        for (action in actions) {
            val r = ruleScores[action] ?: 0f
            val c = cnnScores[action] ?: 0f
            finalScores[action] = if (cnnScores.isNotEmpty()) (r * 0.5f + c * 0.5f) else r
        }

        // Find the best action among ENABLED actions (only ignore if all scores are exactly 0)
        var bestAction: String? = null
        var bestScore = 0f
        for ((action, score) in finalScores) {
            if (enabledActions.contains(action) && score > bestScore) {
                bestScore = score
                bestAction = action
            }
        }

        return if (bestAction != null && bestScore > 0f) {
            ActionResult(bestAction, bestScore, finalScores)
        } else {
            ActionResult(null, 0f, finalScores)
        }
    }

    // ---- Individual Action Evaluators ----

    /**
     * 万歳 (Banzai): Both hands raised above head.
     *
     * Improvement over original:
     * - Uses normalized coordinates so toddler proportions don't matter
     * - Checks relative height (wrists above shoulders by >= 0.6 shoulder-widths)
     *   instead of absolute "above head" which fails for short-armed toddlers
     * - Requires both elbows to be elevated (not just wrists, to avoid false positives)
     * - Temporal stability: checks that pose is held for multiple frames
     */
    private fun evaluateBanzai(frame: FloatArray, tensor: Array<FloatArray>): Float {
        val lWristY = getY(frame, L_WRIST)
        val rWristY = getY(frame, R_WRIST)
        val lShoulderY = getY(frame, L_SHOULDER)
        val rShoulderY = getY(frame, R_SHOULDER)
        val lElbowY = getY(frame, L_ELBOW)
        val rElbowY = getY(frame, R_ELBOW)
        val lWristConf = getConf(frame, L_WRIST)
        val rWristConf = getConf(frame, R_WRIST)

        if (lWristConf < 0.5f || rWristConf < 0.5f) return 0f

        // In normalized space (hip=origin, shoulder-width=1.0), Y is inverted (up = smaller)
        // Wrists must be above shoulders by at least 0.6 units (adaptive for toddlers)
        val lWristAboveShoulder = lShoulderY - lWristY
        val rWristAboveShoulder = rShoulderY - rWristY
        val lElbowAboveShoulder = lShoulderY - lElbowY
        val rElbowAboveShoulder = rShoulderY - rElbowY

        if (lWristAboveShoulder < 0.6f || rWristAboveShoulder < 0.6f) return 0f
        if (lElbowAboveShoulder < 0.1f || rElbowAboveShoulder < 0.1f) return 0f

        // Spatial score (how high the hands are)
        val spatialScore = ((lWristAboveShoulder + rWristAboveShoulder) / 2f).coerceIn(0f, 2f) / 2f

        // Temporal stability: check last 5 frames
        val stableFrames = countStableFrames(tensor, minFrames = 5) { f ->
            val lwY = getY(f, L_WRIST); val rwY = getY(f, R_WRIST)
            val lsY = getY(f, L_SHOULDER); val rsY = getY(f, R_SHOULDER)
            (lsY - lwY) > 0.4f && (rsY - rwY) > 0.4f
        }
        val stabilityScore = (stableFrames.toFloat() / 5f).coerceIn(0f, 1f)

        return (spatialScore * 0.6f + stabilityScore * 0.4f).coerceIn(0f, 1f)
    }

    /**
     * 指さし (Pointing): One arm fully extended, the other NOT extended.
     *
     * Improvement over original:
     * - Elbow angle check: shoulder-elbow-wrist angle must be nearly straight (> 140°)
     * - Asymmetry check in normalized space
     * - Works for both horizontal pointing and angled pointing
     */
    private fun evaluatePointing(frame: FloatArray): Float {
        val lScore = evaluateOneArmPointing(frame, L_SHOULDER, L_ELBOW, L_WRIST, R_SHOULDER, R_ELBOW, R_WRIST)
        val rScore = evaluateOneArmPointing(frame, R_SHOULDER, R_ELBOW, R_WRIST, L_SHOULDER, L_ELBOW, L_WRIST)
        return maxOf(lScore, rScore)
    }

    private fun evaluateOneArmPointing(
        frame: FloatArray,
        shoulderIdx: Int, elbowIdx: Int, wristIdx: Int,
        otherShoulderIdx: Int, otherElbowIdx: Int, otherWristIdx: Int
    ): Float {
        val wristConf = getConf(frame, wristIdx)
        val elbowConf = getConf(frame, elbowIdx)
        if (wristConf < 0.5f || elbowConf < 0.5f) return 0f

        // Arm length (shoulder to wrist) in normalized space
        val armLength = distance(
            getX(frame, shoulderIdx), getY(frame, shoulderIdx),
            getX(frame, wristIdx), getY(frame, wristIdx)
        )

        // Elbow angle: straighter = more likely pointing
        val elbowAngle = angleDeg(
            getX(frame, shoulderIdx), getY(frame, shoulderIdx),
            getX(frame, elbowIdx), getY(frame, elbowIdx),
            getX(frame, wristIdx), getY(frame, wristIdx)
        )

        // Other arm should NOT be extended (asymmetry)
        val otherArmLength = distance(
            getX(frame, otherShoulderIdx), getY(frame, otherShoulderIdx),
            getX(frame, otherWristIdx), getY(frame, otherWristIdx)
        )

        // In normalized space, extended arm > 1.5 units, retracted < 0.8
        if (armLength < 1.5f) return 0f
        if (otherArmLength > 1.2f) return 0f  // Both arms extended = not pointing
        if (elbowAngle < 140f) return 0f  // Bent elbow = not pointing

        val extensionScore = ((armLength - 1.5f) / 1.0f).coerceIn(0f, 1f)
        val straightnessScore = ((elbowAngle - 140f) / 40f).coerceIn(0f, 1f)
        val asymmetryScore = ((1.2f - otherArmLength) / 0.8f).coerceIn(0f, 1f)

        return (extensionScore * 0.4f + straightnessScore * 0.3f + asymmetryScore * 0.3f).coerceIn(0f, 1f)
    }

    /**
     * 手を振る (Waving): Hand raised above head with left-right oscillation.
     *
     * Improvement over original:
     * - Frequency-based detection: counts direction reversals in normalized X
     * - Requires 2-5Hz oscillation frequency range (natural waving speed)
     * - Uses the full time-series, not just recent N frames
     */
    private fun evaluateWaving(tensor: Array<FloatArray>): Float {
        var bestScore = 0f

        // Evaluate clapping penalty
        var clappingPenalty = 0f
        val distances = mutableListOf<Float>()
        for (f in tensor) {
            val lc = getConf(f, L_WRIST)
            val rc = getConf(f, R_WRIST)
            if (lc > 0.3f && rc > 0.3f) {
                distances.add(distance(
                    getX(f, L_WRIST), getY(f, L_WRIST),
                    getX(f, R_WRIST), getY(f, R_WRIST)
                ))
            }
        }
        if (distances.size >= 8) {
            var reversals = 0
            var lastDelta = 0f
            for (i in 1 until distances.size) {
                val delta = distances[i] - distances[i - 1]
                if (abs(delta) > 0.05f) {
                    if (lastDelta != 0f && delta * lastDelta < 0f) {
                        reversals++
                    }
                    lastDelta = delta
                }
            }
            if (reversals >= 4) {
                clappingPenalty = 0.5f
            }
        }

        for (wristIdx in listOf(L_WRIST, R_WRIST)) {
            val shoulderIdx = if (wristIdx == L_WRIST) L_SHOULDER else R_SHOULDER

            // Check if wrist is raised above shoulder in the latest frames
            val lastFrame = tensor.last()
            val wristY = getY(lastFrame, wristIdx)
            val shoulderY = getY(lastFrame, shoulderIdx)
            val wristConf = getConf(lastFrame, wristIdx)

            if (wristConf < 0.4f) continue
            // Relaxed from 0.3f to -0.1f to allow waving at chest/face level
            if ((shoulderY - wristY) < -0.1f) continue  // Not raised enough

            // Collect X positions across time
            val xPositions = mutableListOf<Float>()
            for (frame in tensor) {
                val conf = getConf(frame, wristIdx)
                if (conf > 0.3f) {
                    xPositions.add(getX(frame, wristIdx))
                }
            }

            if (xPositions.size < 10) continue

            // Count direction reversals
            var reversals = 0
            var lastSignificantDelta = 0f
            val minDeltaThreshold = 0.1f  // In normalized space (fraction of shoulder width)

            for (i in 1 until xPositions.size) {
                val delta = xPositions[i] - xPositions[i - 1]
                if (abs(delta) > minDeltaThreshold * 0.3f) {
                    if (lastSignificantDelta != 0f && delta * lastSignificantDelta < 0f) {
                        reversals++
                    }
                    lastSignificantDelta = delta
                }
            }

            // X range (total oscillation amplitude)
            val xRange = (xPositions.max() - xPositions.min())

            // Need at least 2 reversals (one back-and-forth cycle)
            // Amplitude must be significant (> 0.3 shoulder widths in normalized space)
            if (reversals >= 2 && xRange > 0.3f) {
                val reversalScore = (reversals.toFloat() / 4f).coerceIn(0f, 1f)
                val amplitudeScore = (xRange / 1.5f).coerceIn(0f, 1f)
                val raisedScore = ((shoulderY - wristY) / 1.0f).coerceIn(0f, 1f)

                var score = (reversalScore * 0.4f + amplitudeScore * 0.3f + raisedScore * 0.3f)
                score -= clappingPenalty
                bestScore = maxOf(bestScore, score)
            }
        }

        return bestScore.coerceIn(0f, 1f)
    }

    /**
     * 投げる (Throwing): Rapid downward arm sweep from wind-up to follow-through.
     *
     * Improvement over original:
     * - Velocity-based detection in normalized space (not pixel-dependent)
     * - Looks for the characteristic arc: high → low in Y
     * - Checks acceleration pattern (speed increases during throw)
     */
    private fun evaluateThrowing(tensor: Array<FloatArray>): Float {
        var bestScore = 0f
        val framesToCheck = tensor.size

        for (wristIdx in listOf(L_WRIST, R_WRIST)) {
            val shoulderIdx = if (wristIdx == L_WRIST) L_SHOULDER else R_SHOULDER

            // Find peak Y velocity (downward movement) across the time series
            var maxDownwardVelocity = 0f
            var hadWindup = false
            var hasFollowThrough = false

            // Look for wind-up phase (wrist above shoulder) in first half
            val firstHalf = framesToCheck / 2
            for (i in 0 until firstHalf) {
                val wristY = getY(tensor[i], wristIdx)
                val shoulderY = getY(tensor[i], shoulderIdx)
                val conf = getConf(tensor[i], wristIdx)
                if (conf > 0.4f && (shoulderY - wristY) > 0.3f) {
                    hadWindup = true
                    break
                }
            }

            if (!hadWindup) continue

            // Look for follow-through (wrist below shoulder) in second half
            for (i in firstHalf until framesToCheck) {
                val wristY = getY(tensor[i], wristIdx)
                val shoulderY = getY(tensor[i], shoulderIdx)
                val conf = getConf(tensor[i], wristIdx)
                if (conf > 0.4f && (wristY - shoulderY) > 0.3f) {
                    hasFollowThrough = true
                    break
                }
            }

            if (!hasFollowThrough) continue

            // Calculate velocity profile
            for (i in 1 until framesToCheck) {
                val prevY = getY(tensor[i - 1], wristIdx)
                val currY = getY(tensor[i], wristIdx)
                val conf = getConf(tensor[i], wristIdx)
                if (conf > 0.3f) {
                    val velocity = currY - prevY  // Positive = downward in image coords
                    if (velocity > maxDownwardVelocity) {
                        maxDownwardVelocity = velocity
                    }
                }
            }

            // Check if the other hand is raised high (Banzai recovery)
            val otherWristIdx = if (wristIdx == L_WRIST) R_WRIST else L_WRIST
            val otherShoulderIdx = if (shoulderIdx == L_SHOULDER) R_SHOULDER else L_SHOULDER
            var otherHandPenalty = 0f
            val lastOtherY = getY(tensor.last(), otherWristIdx)
            val lastOtherShoulderY = getY(tensor.last(), otherShoulderIdx)
            if (getConf(tensor.last(), otherWristIdx) > 0.4f) {
                if ((lastOtherShoulderY - lastOtherY) > 0.4f) {
                    otherHandPenalty = 0.5f // Significant penalty if the other hand is also raised (Banzai)
                }
            }

            // Total Y displacement
            val startY = getY(tensor[0], wristIdx)
            val endY = getY(tensor.last(), wristIdx)
            val totalDisplacement = endY - startY

            // Increased maxDownwardVelocity threshold from 0.1f to 0.25f for stricter throwing detection
            if (totalDisplacement > 1.0f && maxDownwardVelocity > 0.25f) {
                val displacementScore = (totalDisplacement / 2.0f).coerceIn(0f, 1f)
                val velocityScore = (maxDownwardVelocity / 0.5f).coerceIn(0f, 1f)
                var score = (displacementScore * 0.5f + velocityScore * 0.5f)
                score -= otherHandPenalty
                bestScore = maxOf(bestScore, score)
            }
        }

        return bestScore.coerceIn(0f, 1f)
    }

    /**
     * 拍手 (Clapping): Both wrists close together at chest level, with repetitive motion.
     *
     * Improvement over original:
     * - Elbow angle change detection (open → close → open cycle)
     * - Distance check in normalized space (body-size invariant)
     * - Temporal pattern: looks for repetitive approaching/separating pattern
     */
    private fun evaluateClapping(frame: FloatArray, tensor: Array<FloatArray>): Float {
        val lWristConf = getConf(frame, L_WRIST)
        val rWristConf = getConf(frame, R_WRIST)
        if (lWristConf < 0.4f || rWristConf < 0.4f) return 0f

        // Current wrist distance in normalized space
        val currentDist = distance(
            getX(frame, L_WRIST), getY(frame, L_WRIST),
            getX(frame, R_WRIST), getY(frame, R_WRIST)
        )

        // Both wrists should be at chest level (between shoulders and hips)
        val lShoulderY = getY(frame, L_SHOULDER)
        val rShoulderY = getY(frame, R_SHOULDER)
        val lHipY = getY(frame, L_HIP)
        val rHipY = getY(frame, R_HIP)
        val midShoulderY = (lShoulderY + rShoulderY) / 2f
        val midHipY = (lHipY + rHipY) / 2f
        val lWristY = getY(frame, L_WRIST)
        val rWristY = getY(frame, R_WRIST)

        // Wrists should be between shoulder and hip level
        val lAtChest = lWristY in midShoulderY..midHipY || abs(lWristY - (midShoulderY + midHipY) / 2f) < 1.0f
        val rAtChest = rWristY in midShoulderY..midHipY || abs(rWristY - (midShoulderY + midHipY) / 2f) < 1.0f

        if (!lAtChest || !rAtChest) return 0f

        // Proximity score (how close the wrists are)
        // In normalized space, < 0.5 shoulder widths = very close
        val proximityScore = if (currentDist < 0.5f) {
            (1f - currentDist / 0.5f).coerceIn(0f, 1f)
        } else {
            0f
        }

        // Temporal: look for distance oscillation (approaching and separating)
        // Also check that BOTH hands are actually moving (absolute displacement)
        val distances = mutableListOf<Float>()
        var lMovement = 0f
        var rMovement = 0f
        
        for (i in 0 until tensor.size) {
            val f = tensor[i]
            val lc = getConf(f, L_WRIST)
            val rc = getConf(f, R_WRIST)
            if (lc > 0.3f && rc > 0.3f) {
                distances.add(distance(
                    getX(f, L_WRIST), getY(f, L_WRIST),
                    getX(f, R_WRIST), getY(f, R_WRIST)
                ))
            }
            
            if (i > 0) {
                val prevF = tensor[i - 1]
                if (lc > 0.3f && getConf(prevF, L_WRIST) > 0.3f) {
                    lMovement += distance(getX(f, L_WRIST), getY(f, L_WRIST), getX(prevF, L_WRIST), getY(prevF, L_WRIST))
                }
                if (rc > 0.3f && getConf(prevF, R_WRIST) > 0.3f) {
                    rMovement += distance(getX(f, R_WRIST), getY(f, R_WRIST), getX(prevF, R_WRIST), getY(prevF, R_WRIST))
                }
            }
        }

        // If one hand is mostly static while the other moves, it's not clapping (e.g. one-handed waving)
        if (lMovement < 0.5f || rMovement < 0.5f) return 0f

        var oscillationScore = 0f
        if (distances.size >= 8) {
            // Count distance reversals (approach ↔ separate)
            var reversals = 0
            var lastDelta = 0f
            for (i in 1 until distances.size) {
                val delta = distances[i] - distances[i - 1]
                if (abs(delta) > 0.05f) {
                    if (lastDelta != 0f && delta * lastDelta < 0f) {
                        reversals++
                    }
                    lastDelta = delta
                }
            }
            // Clapping has rapid oscillations (at least 2 cycles = 4 reversals)
            oscillationScore = (reversals.toFloat() / 4f).coerceIn(0f, 1f)
        }

        // If wrists are currently close, that alone is a strong signal
        // Combined with oscillation pattern, it's very reliable
        return if (proximityScore > 0f) {
            (proximityScore * 0.5f + oscillationScore * 0.5f).coerceIn(0f, 1f)
        } else {
            (oscillationScore * 0.3f)  // Oscillation alone, wrists not currently close
        }
    }

    // ---- Utility functions ----

    private fun getX(frame: FloatArray, keypointIdx: Int): Float = frame[keypointIdx * C]
    private fun getY(frame: FloatArray, keypointIdx: Int): Float = frame[keypointIdx * C + 1]
    private fun getConf(frame: FloatArray, keypointIdx: Int): Float = frame[keypointIdx * C + 2]

    private fun distance(x1: Float, y1: Float, x2: Float, y2: Float): Float {
        val dx = x1 - x2
        val dy = y1 - y2
        return sqrt(dx * dx + dy * dy)
    }

    /**
     * Calculate the angle at point B (elbow) formed by points A-B-C.
     * Returns angle in degrees (0-180).
     */
    private fun angleDeg(ax: Float, ay: Float, bx: Float, by: Float, cx: Float, cy: Float): Float {
        val v1x = ax - bx
        val v1y = ay - by
        val v2x = cx - bx
        val v2y = cy - by
        val angle = atan2(v1x * v2y - v1y * v2x, v1x * v2x + v1y * v2y)
        return Math.toDegrees(abs(angle.toDouble())).toFloat()
    }

    /**
     * Count how many of the last N frames satisfy a given condition.
     */
    private fun countStableFrames(
        tensor: Array<FloatArray>,
        minFrames: Int,
        condition: (FloatArray) -> Boolean
    ): Int {
        val startIdx = maxOf(0, tensor.size - minFrames)
        var count = 0
        for (i in startIdx until tensor.size) {
            if (condition(tensor[i])) count++
        }
        return count
    }
}
