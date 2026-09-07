package com.example.director

import android.graphics.Bitmap
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.Rect
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.sqrt

object ScoringEngine {

    fun calculateScore(bitmap: Bitmap, detection: DetectionResult, config: DirectorConfig): Float {
        // 1. Composition Score (position and size)
        val w = bitmap.width.toFloat()
        val h = bitmap.height.toFloat()
        
        // Detection is in relative coordinates (0.0 to 1.0)
        val cx = (detection.x1 + detection.x2) / 2f
        val cy = (detection.y1 + detection.y2) / 2f
        val areaRatio = (detection.x2 - detection.x1) * (detection.y2 - detection.y1)
        
        val targetCx = config.targetComposition.subjectCenterX
        val targetCy = config.targetComposition.subjectCenterY
        val targetArea = config.targetComposition.subjectAreaRatio
        
        // Distance penalty
        val dist = sqrt((cx - targetCx) * (cx - targetCx) + (cy - targetCy) * (cy - targetCy))
        var compScore = max(0f, 100f - (dist * 200f)) // simplified penalty
        
        // Area penalty
        val areaDiff = abs(areaRatio - targetArea)
        compScore -= areaDiff * 100f
        compScore = max(0f, compScore)

        // 2. Lighting Score (HSV brightness calculation via OpenCV)
        // Convert Bitmap to OpenCV Mat
        val mat = Mat()
        Utils.bitmapToMat(bitmap, mat)
        
        // Convert to HSV
        val hsvMat = Mat()
        Imgproc.cvtColor(mat, hsvMat, Imgproc.COLOR_RGB2HSV)

        // Calculate Bounding Box in pixel coordinates
        val px1 = (detection.x1 * w).toInt().coerceIn(0, w.toInt() - 1)
        val py1 = (detection.y1 * h).toInt().coerceIn(0, h.toInt() - 1)
        val px2 = (detection.x2 * w).toInt().coerceIn(0, w.toInt() - 1)
        val py2 = (detection.y2 * h).toInt().coerceIn(0, h.toInt() - 1)
        
        val rect = Rect(px1, py1, px2 - px1, py2 - py1)
        var lightScore = 100f
        
        if (rect.width > 0 && rect.height > 0) {
            // Extract Subject Mat
            val subjectMat = hsvMat.submat(rect)
            
            // Calculate median V (Brightness) for subject
            // For PoC, we use mean instead of median for simplicity and speed
            val subjectMean = Core.mean(subjectMat)
            val subjectV = subjectMean.`val`[2] // HSV, V is index 2

            // Background V (simplified: sample top-left 100x100 box if available)
            val bgRect = Rect(0, 0, (w * 0.2).toInt(), (h * 0.2).toInt())
            val bgMat = hsvMat.submat(bgRect)
            val bgMean = Core.mean(bgMat)
            val bgV = bgMean.`val`[2]
            
            val targetSubjV = config.targetLighting.subjectBrightnessTarget
            val targetBgV = config.targetLighting.backgroundBrightnessTarget
            
            val vDiff = abs(subjectV - targetSubjV).toFloat()
            val bgDiff = abs(bgV - targetBgV).toFloat()
            
            lightScore = max(0f, 100f - vDiff - (bgDiff * 0.5f))
            
            subjectMat.release()
            bgMat.release()
        }
        
        hsvMat.release()
        mat.release()
        
        // Combine scores (70% composition, 30% lighting)
        return (compScore * 0.7f) + (lightScore * 0.3f)
    }
}
