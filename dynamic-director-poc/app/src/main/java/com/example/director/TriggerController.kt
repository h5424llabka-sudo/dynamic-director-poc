package com.example.director

/**
 * Events emitted by the trigger controller.
 */
sealed class TriggerEvent {
    /**
     * Fire the shutter. Contains the trigger reason and confidence.
     */
    data class Fire(val reason: String, val confidence: Float) : TriggerEvent()

    /**
     * Currently in cooldown period after a recent capture.
     */
    data class Cooldown(val remainingMs: Long) : TriggerEvent()

    /**
     * Idle — no trigger conditions met.
     */
    data class Idle(val status: String, val scores: Map<String, Float>) : TriggerEvent()
}

/**
 * Trigger controller that prevents false positives and jitter in auto-shutter decisions.
 *
 * Implements three key mechanisms:
 * 1. **Hysteresis thresholds**: Different thresholds for activation (high) and deactivation (low)
 *    to prevent "flickering" on/off when scores hover around a single threshold.
 * 2. **Confirmation window**: Requires the action to be detected for multiple consecutive frames
 *    before firing, filtering out momentary false positives.
 * 3. **Cooldown management**: Prevents photo spam after a successful capture.
 *
 * The controller is stateful and should be called once per frame.
 */
class TriggerController(
    private val activationThresholds: Map<String, Float> = emptyMap(),
    private val confirmationFrames: Int = 3,
    private val cooldownMs: Long = 3000L,
    private val smileThreshold: Float = 0.80f
) {
    // State tracking
    private var isActive = false  // Has the activation threshold been crossed?
    private var activeActionName: String? = null
    private var consecutiveActiveFrames = 0
    private var lastCaptureTime = 0L

    /**
     * Update the trigger state with the latest classification results.
     * Should be called once per analyzed frame.
     *
     * @param actionResult The action classification result (from ActionClassifier)
     * @param faceScore The face/smile score (0-100 from ScoringEngine)
     * @param smileProbability Raw smile probability (0.0-1.0 from FaceAnalyzer)
     * @return TriggerEvent indicating what action to take
     */
    fun update(
        actionResult: ActionResult?,
        faceScore: Float,
        smileProbability: Float,
        referenceScore: Float = 0f
    ): TriggerEvent {
        val now = System.currentTimeMillis()

        // Check cooldown
        val timeSinceCapture = now - lastCaptureTime
        if (lastCaptureTime > 0 && timeSinceCapture < cooldownMs) {
            // Reset state during cooldown
            isActive = false
            consecutiveActiveFrames = 0
            activeActionName = null
            return TriggerEvent.Cooldown(cooldownMs - timeSinceCapture)
        }

        // Evaluate pose trigger (with hysteresis)
        val poseTriggered = evaluatePoseTrigger(actionResult)

        // Evaluate smile trigger (simpler: just check threshold with confirmation)
        val smileTriggered = smileProbability >= smileThreshold
        
        // Evaluate reference composition trigger
        val referenceTriggered = referenceScore >= 0.8f

        // Build status info
        val scores = mutableMapOf<String, Float>()
        actionResult?.rawScores?.let { scores.putAll(it) }
        scores["Smile"] = smileProbability * 100f
        scores["RefMatch"] = referenceScore * 100f

        return when {
            poseTriggered -> {
                val reason = "Action: ${activeActionName ?: actionResult?.actionName ?: "Unknown"}"
                val confidence = actionResult?.confidence ?: 0f
                lastCaptureTime = now
                resetState()
                TriggerEvent.Fire(reason, confidence)
            }
            smileTriggered -> {
                // Smile also needs confirmation (prevent false triggers from brief expressions)
                consecutiveActiveFrames++
                if (consecutiveActiveFrames >= confirmationFrames) {
                    val reason = "Smile: ${(smileProbability * 100).toInt()}%"
                    lastCaptureTime = now
                    resetState()
                    TriggerEvent.Fire(reason, smileProbability)
                } else {
                    TriggerEvent.Idle("Smile confirming (${consecutiveActiveFrames}/$confirmationFrames)", scores)
                }
            }
            referenceTriggered -> {
                consecutiveActiveFrames++
                if (consecutiveActiveFrames >= confirmationFrames) {
                    val reason = "Composition Match: ${(referenceScore * 100).toInt()}%"
                    lastCaptureTime = now
                    resetState()
                    TriggerEvent.Fire(reason, referenceScore)
                } else {
                    TriggerEvent.Idle("Composition confirming (${consecutiveActiveFrames}/$confirmationFrames)", scores)
                }
            }
            else -> {
                // Neither triggered
                if (!isActive) {
                    consecutiveActiveFrames = 0
                }
                val status = if (actionResult?.actionName != null) {
                    "Detected: ${actionResult.actionName} (${(actionResult.confidence * 100).toInt()}%)"
                } else {
                    "Searching..."
                }
                TriggerEvent.Idle(status, scores)
            }
        }
    }

    /**
     * Manually reset cooldown (e.g., when user changes settings).
     */
    fun resetCooldown() {
        lastCaptureTime = 0L
    }

    /**
     * Update configuration at runtime.
     */
    fun updateConfig(
        newActivationThresholds: Map<String, Float>? = null,
        newConfirmationFrames: Int? = null,
        newCooldownMs: Long? = null,
        newSmileThreshold: Float? = null
    ): TriggerController {
        return TriggerController(
            activationThresholds = newActivationThresholds ?: activationThresholds,
            confirmationFrames = newConfirmationFrames ?: confirmationFrames,
            cooldownMs = newCooldownMs ?: cooldownMs,
            smileThreshold = newSmileThreshold ?: smileThreshold
        )
    }

    // ---- Private helpers ----

    /**
     * Evaluate pose trigger with hysteresis logic.
     *
     * When inactive: require score > activationThreshold to become active
     * When active: stay active as long as score > deactivationThreshold
     * Fire after confirmationFrames consecutive active frames
     */
    private fun evaluatePoseTrigger(actionResult: ActionResult?): Boolean {
        val actionName = actionResult?.actionName
        val confidence = actionResult?.confidence ?: 0f

        if (!isActive) {
            // Not yet active — need to cross the activation threshold
            val threshold = if (actionName != null) activationThresholds[actionName] ?: 0.70f else 0.70f
            if (actionName != null && confidence >= threshold) {
                isActive = true
                activeActionName = actionName
                consecutiveActiveFrames = 1
            }
            return false
        } else {
            // Already active — check if we should deactivate or confirm
            val currentActivationThreshold = activationThresholds[activeActionName] ?: 0.70f
            val deactivationThreshold = currentActivationThreshold * 0.6f
            
            if (actionName == null || confidence < deactivationThreshold) {
                // Dropped below deactivation threshold — reset
                resetState()
                return false
            }

            // Action changed — reset confirmation counter for the new action
            if (actionName != activeActionName) {
                val newThreshold = activationThresholds[actionName] ?: 0.70f
                if (confidence >= newThreshold) {
                    activeActionName = actionName
                    consecutiveActiveFrames = 1
                } else {
                    resetState()
                }
                return false
            }

            // Same action, still above deactivation threshold
            consecutiveActiveFrames++
            return consecutiveActiveFrames >= confirmationFrames
        }
    }

    private fun resetState() {
        isActive = false
        activeActionName = null
        consecutiveActiveFrames = 0
    }
}
