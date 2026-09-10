package com.example.director

import android.graphics.Bitmap
import android.util.Log
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.segmentation.subject.SubjectSegmentation
import com.google.mlkit.vision.segmentation.subject.SubjectSegmenterOptions
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.nio.FloatBuffer

/**
 * Subject Segmentation Engine that isolates subjects (e.g., toddlers) and
 * applies a background blur (bokeh) effect.
 */
class SegmentationEngine {

    private val segmenter = SubjectSegmentation.getClient(
        SubjectSegmenterOptions.Builder()
            .enableForegroundConfidenceMask()
            .build()
    )

    /**
     * Process a bitmap to apply background blur.
     * 
     * @param originalBitmap The input image to process
     * @param onSuccess Callback with the blurred result bitmap
     * @param onFailure Callback on error
     */
    fun applyBokehEffect(
        originalBitmap: Bitmap,
        onSuccess: (Bitmap) -> Unit,
        onFailure: (Exception) -> Unit
    ) {
        val inputImage = InputImage.fromBitmap(originalBitmap, 0)
        
        segmenter.process(inputImage)
            .addOnSuccessListener { result ->
                val mask = result.foregroundConfidenceMask
                if (mask == null) {
                    onSuccess(originalBitmap)
                    return@addOnSuccessListener
                }

                // Run image processing on a background thread
                java.util.concurrent.Executors.newSingleThreadExecutor().execute {
                    try {
                        val resultBitmap = processBlurWithOpenCV(originalBitmap, mask, originalBitmap.width, originalBitmap.height)
                        onSuccess(resultBitmap)
                    } catch (e: Exception) {
                        onFailure(e)
                    }
                }
            }
            .addOnFailureListener { e ->
                onFailure(e)
            }
    }

    private fun processBlurWithOpenCV(
        original: Bitmap,
        maskBuffer: FloatBuffer,
        maskWidth: Int,
        maskHeight: Int
    ): Bitmap {
        // 1. Convert original Bitmap to OpenCV Mat (RGBA)
        val origMat = Mat()
        Utils.bitmapToMat(original, origMat)
        
        // Convert to RGB for processing
        val rgbMat = Mat()
        Imgproc.cvtColor(origMat, rgbMat, Imgproc.COLOR_RGBA2RGB)
        
        // 2. Create Blurred Background
        val blurredMat = Mat()
        Imgproc.GaussianBlur(rgbMat, blurredMat, Size(55.0, 55.0), 0.0) // heavy blur
        
        // 3. Construct Mask Mat from FloatBuffer
        maskBuffer.rewind()
        val maskData = FloatArray(maskBuffer.remaining())
        maskBuffer.get(maskData)
        
        val maskMatF = Mat(maskHeight, maskWidth, CvType.CV_32FC1)
        maskMatF.put(0, 0, maskData)
        
        // Resize mask to match original image size if necessary
        val resizedMask = Mat()
        if (maskWidth != origMat.cols() || maskHeight != origMat.rows()) {
            Imgproc.resize(maskMatF, resizedMask, Size(origMat.cols().toDouble(), origMat.rows().toDouble()))
        } else {
            maskMatF.copyTo(resizedMask)
        }
        
        // OpenCV blend: output = original * mask + blurred * (1 - mask)
        val mask3c = Mat()
        val maskChannels = listOf(resizedMask, resizedMask, resizedMask)
        Core.merge(maskChannels, mask3c)
        
        val invertedMask3c = Mat()
        Core.subtract(Mat.ones(mask3c.size(), mask3c.type()), mask3c, invertedMask3c)
        
        val origFloat = Mat()
        rgbMat.convertTo(origFloat, CvType.CV_32FC3)
        
        val blurFloat = Mat()
        blurredMat.convertTo(blurFloat, CvType.CV_32FC3)
        
        val fg = Mat()
        Core.multiply(origFloat, mask3c, fg)
        
        val bg = Mat()
        Core.multiply(blurFloat, invertedMask3c, bg)
        
        val resultFloat = Mat()
        Core.add(fg, bg, resultFloat)
        
        // Convert back to 8-bit RGB
        val finalRgb = Mat()
        resultFloat.convertTo(finalRgb, CvType.CV_8UC3)
        
        // Convert back to RGBA for Android Bitmap
        val finalRgba = Mat()
        Imgproc.cvtColor(finalRgb, finalRgba, Imgproc.COLOR_RGB2RGBA)
        
        val resultBitmap = Bitmap.createBitmap(original.width, original.height, Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(finalRgba, resultBitmap)
        
        // Release Mat objects
        origMat.release()
        rgbMat.release()
        blurredMat.release()
        maskMatF.release()
        resizedMask.release()
        mask3c.release()
        invertedMask3c.release()
        origFloat.release()
        blurFloat.release()
        fg.release()
        bg.release()
        resultFloat.release()
        finalRgb.release()
        finalRgba.release()
        
        return resultBitmap
    }
}
