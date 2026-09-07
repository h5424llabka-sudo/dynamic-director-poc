package com.example.director

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.nio.MappedByteBuffer

data class DetectionResult(
    val x1: Float,
    val y1: Float,
    val x2: Float,
    val y2: Float,
    val confidence: Float,
    val classId: Int
)

class YoloDetector(context: Context, modelName: String = "yolov8n_float16.tflite") {
    private var interpreter: Interpreter? = null
    private val imageSize = 640 // standard for YOLOv8n

    init {
        try {
            val model: MappedByteBuffer = FileUtil.loadMappedFile(context, modelName)
            val options = Interpreter.Options()
            options.numThreads = 4
            interpreter = Interpreter(model, options)
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    fun detect(bitmap: Bitmap): DetectionResult? {
        val tflite = interpreter ?: return null

        // Prepare input image
        val imageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(imageSize, imageSize, ResizeOp.ResizeMethod.BILINEAR))
            .build()
        var tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(bitmap)
        tensorImage = imageProcessor.process(tensorImage)

        // Normalize image if required (YOLOv8 expects 0.0-1.0 float values)
        val inputBuffer = tensorImage.tensorBuffer.buffer
        val floatArray = FloatArray(imageSize * imageSize * 3)
        inputBuffer.rewind()
        for (i in 0 until inputBuffer.capacity() / 4) {
            floatArray[i] = inputBuffer.float / 255.0f
        }
        val finalInput = TensorBuffer.createFixedSize(intArrayOf(1, imageSize, imageSize, 3), DataType.FLOAT32)
        finalInput.loadArray(floatArray)

        // YOLOv8 output is [1, 84, 8400]
        val outputBuffer = TensorBuffer.createFixedSize(intArrayOf(1, 84, 8400), DataType.FLOAT32)
        
        try {
            tflite.run(finalInput.buffer, outputBuffer.buffer)
        } catch (e: Exception) {
            e.printStackTrace()
            return null
        }

        val outputArray = outputBuffer.floatArray
        
        // Simplified Parsing for PoC: Find the best Person (class 0) detection
        var bestConf = 0f
        var bestBox: DetectionResult? = null

        // 8400 anchor boxes, 84 values each (cx, cy, w, h, 80 class scores)
        // Memory layout of [1, 84, 8400] means for each of the 84 properties, we have 8400 values.
        for (i in 0 until 8400) {
            // class 0 score is at index 4 in the 84 properties block
            // Indexing for [1, 84, 8400]: flatten_index = prop_idx * 8400 + anchor_idx
            val personConf = outputArray[4 * 8400 + i]
            
            if (personConf > 0.5f && personConf > bestConf) {
                bestConf = personConf
                val cx = outputArray[0 * 8400 + i]
                val cy = outputArray[1 * 8400 + i]
                val w = outputArray[2 * 8400 + i]
                val h = outputArray[3 * 8400 + i]
                
                // Convert back to original image scale relative coordinates (0.0 - 1.0)
                val x1 = (cx - w / 2f) / imageSize.toFloat()
                val y1 = (cy - h / 2f) / imageSize.toFloat()
                val x2 = (cx + w / 2f) / imageSize.toFloat()
                val y2 = (cy + h / 2f) / imageSize.toFloat()
                
                bestBox = DetectionResult(x1, y1, x2, y2, personConf, 0)
            }
        }
        
        return bestBox
    }

    fun close() {
        interpreter?.close()
        interpreter = null
    }
}
