package com.example.director

import com.google.mlkit.vision.pose.Pose
import com.google.mlkit.vision.pose.PoseLandmark

/**
 * Timestamped skeleton data for a single frame.
 * Stores normalized keypoint coordinates with confidence scores.
 */
data class TimestampedPose(
    val timestamp: Long,  // System.nanoTime()
    val keypoints: FloatArray,  // [x, y, confidence] × 9 points = 27 values
    val isValid: Boolean  // true if enough keypoints had sufficient confidence
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is TimestampedPose) return false
        return timestamp == other.timestamp && keypoints.contentEquals(other.keypoints) && isValid == other.isValid
    }
    override fun hashCode(): Int = timestamp.hashCode()
}

/**
 * Time-series ring buffer with equal-time resampling for FPS-invariant action recognition.
 *
 * This buffer solves the core problem of variable frame rates on mobile devices:
 * - Stores raw skeleton data with timestamps in a ring buffer (past T seconds)
 * - Resamples to fixed-count frames via linear interpolation
 * - Normalizes coordinates relative to body proportions (shoulder width as unit length)
 * - Compensates for dropped frames and low-confidence keypoints
 *
 * Output tensor shape: [outputFrames, 27] where 27 = 9 keypoints × 3 (x, y, confidence)
 */
class PoseTimeSeriesBuffer(
    private val windowSeconds: Float = 1.0f,
    private val outputFrames: Int = 30,
    private val minConfidence: Float = 0.5f
) {
    companion object {
        // Upper-body keypoints most relevant for toddler action recognition
        // Order: Nose, L-Shoulder, R-Shoulder, L-Elbow, R-Elbow, L-Wrist, R-Wrist, L-Hip, R-Hip
        val KEYPOINT_IDS = intArrayOf(
            PoseLandmark.NOSE,
            PoseLandmark.LEFT_SHOULDER,
            PoseLandmark.RIGHT_SHOULDER,
            PoseLandmark.LEFT_ELBOW,
            PoseLandmark.RIGHT_ELBOW,
            PoseLandmark.LEFT_WRIST,
            PoseLandmark.RIGHT_WRIST,
            PoseLandmark.LEFT_HIP,
            PoseLandmark.RIGHT_HIP
        )
        const val NUM_KEYPOINTS = 9
        const val CHANNELS_PER_KEYPOINT = 3  // x, y, confidence
        const val FEATURE_DIM = NUM_KEYPOINTS * CHANNELS_PER_KEYPOINT  // 27
    }

    private val windowNanos = (windowSeconds * 1_000_000_000L).toLong()
    private val rawBuffer = ArrayDeque<TimestampedPose>(120)  // ~4s at 30fps

    // Cached shoulder width for normalization (EMA smoothed)
    private var smoothedShoulderWidth = 0f
    private val shoulderWidthAlpha = 0.3f

    /**
     * Add a new ML Kit Pose frame to the buffer.
     * Extracts relevant keypoints and stores with timestamp.
     */
    fun addFrame(pose: Pose, imageWidth: Int, imageHeight: Int) {
        val now = System.nanoTime()
        val keypoints = FloatArray(FEATURE_DIM)
        var validCount = 0

        for (i in KEYPOINT_IDS.indices) {
            val landmark = pose.getPoseLandmark(KEYPOINT_IDS[i])
            val baseIdx = i * CHANNELS_PER_KEYPOINT
            if (landmark != null && landmark.inFrameLikelihood > minConfidence) {
                // Store as relative coordinates (0.0 - 1.0)
                keypoints[baseIdx] = landmark.position.x / imageWidth.toFloat()
                keypoints[baseIdx + 1] = landmark.position.y / imageHeight.toFloat()
                keypoints[baseIdx + 2] = landmark.inFrameLikelihood
                validCount++
            } else {
                // Mark as invalid (will be interpolated later)
                keypoints[baseIdx] = Float.NaN
                keypoints[baseIdx + 1] = Float.NaN
                keypoints[baseIdx + 2] = 0f
            }
        }

        val isValid = validCount >= 5  // Need at least 5 of 9 keypoints

        rawBuffer.addLast(TimestampedPose(now, keypoints, isValid))

        // Prune old entries beyond the window
        val cutoff = now - windowNanos * 2  // Keep 2x window for interpolation margin
        while (rawBuffer.isNotEmpty() && rawBuffer.first().timestamp < cutoff) {
            rawBuffer.removeFirst()
        }

        // Update smoothed shoulder width
        updateShoulderWidth(pose, imageWidth)
    }

    /**
     * Generate a fixed-size, time-normalized, spatially-normalized tensor.
     * Returns null if insufficient data is available.
     *
     * Output shape: [outputFrames][FEATURE_DIM]
     */
    fun getResampledTensor(): Array<FloatArray>? {
        if (rawBuffer.size < 3) return null

        val now = System.nanoTime()
        val windowStart = now - windowNanos

        // Filter to frames within the time window
        val windowedFrames = rawBuffer.filter { it.timestamp >= windowStart }
        if (windowedFrames.size < 2) return null

        // Generate equal-time grid
        val result = Array(outputFrames) { FloatArray(FEATURE_DIM) }
        val timeStep = windowNanos.toDouble() / (outputFrames - 1)

        for (frameIdx in 0 until outputFrames) {
            val targetTime = windowStart + (frameIdx * timeStep).toLong()
            val interpolated = interpolateAtTime(targetTime, windowedFrames)
            result[frameIdx] = interpolated
        }

        // Spatial normalization
        normalizeInPlace(result)

        return result
    }

    /**
     * Get the current shoulder width (smoothed), useful for legacy rule-based checks.
     */
    fun getSmoothedShoulderWidth(): Float = smoothedShoulderWidth

    /**
     * Check if we have enough data for reliable classification.
     */
    fun isReady(): Boolean {
        if (rawBuffer.size < 5) return false
        val now = System.nanoTime()
        val windowStart = now - windowNanos
        return rawBuffer.count { it.timestamp >= windowStart } >= 3
    }

    /**
     * Clear all buffered data.
     */
    fun clear() {
        rawBuffer.clear()
        smoothedShoulderWidth = 0f
    }

    // ---- Private helpers ----

    /**
     * Linear interpolation of keypoints at a specific timestamp.
     * Finds the two nearest frames (before and after) and lerps between them.
     */
    private fun interpolateAtTime(targetTime: Long, frames: List<TimestampedPose>): FloatArray {
        // Find the bracketing frames
        var beforeIdx = -1
        var afterIdx = -1

        for (i in frames.indices) {
            if (frames[i].timestamp <= targetTime) {
                beforeIdx = i
            }
            if (frames[i].timestamp >= targetTime && afterIdx == -1) {
                afterIdx = i
            }
        }

        // Edge cases: target is outside the range
        if (beforeIdx == -1) return compensateDroppedFrame(frames.first().keypoints)
        if (afterIdx == -1) return compensateDroppedFrame(frames.last().keypoints)
        if (beforeIdx == afterIdx) return compensateDroppedFrame(frames[beforeIdx].keypoints)

        val before = frames[beforeIdx]
        val after = frames[afterIdx]
        val timeDelta = (after.timestamp - before.timestamp).toFloat()
        val t = if (timeDelta > 0f) {
            ((targetTime - before.timestamp).toFloat() / timeDelta).coerceIn(0f, 1f)
        } else {
            0.5f
        }

        val result = FloatArray(FEATURE_DIM)
        for (i in 0 until FEATURE_DIM) {
            val a = before.keypoints[i]
            val b = after.keypoints[i]
            result[i] = when {
                a.isNaN() && b.isNaN() -> Float.NaN
                a.isNaN() -> b
                b.isNaN() -> a
                else -> a + (b - a) * t
            }
        }
        return result
    }

    /**
     * Compensate for dropped/invalid keypoints by replacing NaN values
     * with nearest valid neighbor values.
     */
    private fun compensateDroppedFrame(keypoints: FloatArray): FloatArray {
        val result = keypoints.copyOf()
        for (i in result.indices) {
            if (result[i].isNaN()) {
                // Try to find the last valid value from previous frames
                result[i] = findLastValidValue(i) ?: 0.5f  // Default to center
            }
        }
        return result
    }

    /**
     * Search backward through the buffer for the most recent valid value
     * at a specific keypoint index.
     */
    private fun findLastValidValue(index: Int): Float? {
        for (i in rawBuffer.indices.reversed()) {
            val v = rawBuffer.elementAt(i).keypoints[index]
            if (!v.isNaN()) return v
        }
        return null
    }

    /**
     * Spatial normalization in-place:
     * 1. Translate so hip center = (0, 0)
     * 2. Scale so shoulder width = 1.0
     */
    private fun normalizeInPlace(tensor: Array<FloatArray>) {
        if (smoothedShoulderWidth <= 0.001f) return  // Can't normalize yet

        for (frame in tensor) {
            // Compute hip center for this frame (keypoints 7=L-Hip, 8=R-Hip)
            val lHipX = frame[7 * CHANNELS_PER_KEYPOINT]
            val lHipY = frame[7 * CHANNELS_PER_KEYPOINT + 1]
            val rHipX = frame[8 * CHANNELS_PER_KEYPOINT]
            val rHipY = frame[8 * CHANNELS_PER_KEYPOINT + 1]

            val hipCenterX = if (!lHipX.isNaN() && !rHipX.isNaN()) (lHipX + rHipX) / 2f else 0.5f
            val hipCenterY = if (!lHipY.isNaN() && !rHipY.isNaN()) (lHipY + rHipY) / 2f else 0.5f

            // Translate and scale each keypoint
            for (kp in 0 until NUM_KEYPOINTS) {
                val baseIdx = kp * CHANNELS_PER_KEYPOINT
                if (!frame[baseIdx].isNaN()) {
                    frame[baseIdx] = (frame[baseIdx] - hipCenterX) / smoothedShoulderWidth
                    frame[baseIdx + 1] = (frame[baseIdx + 1] - hipCenterY) / smoothedShoulderWidth
                    // Confidence (index +2) stays as-is
                }
            }
        }
    }

    /**
     * Update the smoothed shoulder width using exponential moving average.
     */
    private fun updateShoulderWidth(pose: Pose, imageWidth: Int) {
        val lShoulder = pose.getPoseLandmark(PoseLandmark.LEFT_SHOULDER)
        val rShoulder = pose.getPoseLandmark(PoseLandmark.RIGHT_SHOULDER)
        if (lShoulder != null && rShoulder != null
            && lShoulder.inFrameLikelihood > minConfidence
            && rShoulder.inFrameLikelihood > minConfidence) {
            val dx = (lShoulder.position.x - rShoulder.position.x) / imageWidth.toFloat()
            val dy = (lShoulder.position.y - rShoulder.position.y) / imageWidth.toFloat()
            val currentWidth = kotlin.math.sqrt(dx * dx + dy * dy)
            if (currentWidth > 0.01f) {
                smoothedShoulderWidth = if (smoothedShoulderWidth <= 0.001f) {
                    currentWidth
                } else {
                    smoothedShoulderWidth * (1 - shoulderWidthAlpha) + currentWidth * shoulderWidthAlpha
                }
            }
        }
    }
}
