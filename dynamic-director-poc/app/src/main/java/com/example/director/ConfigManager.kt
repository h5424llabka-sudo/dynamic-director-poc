package com.example.director

import android.content.Context
import org.json.JSONObject
import java.io.BufferedReader

data class TargetComposition(
    val subjectCenterX: Float,
    val subjectCenterY: Float,
    val subjectAreaRatio: Float
)

data class TargetLighting(
    val isBacklit: Boolean,
    val subjectBrightnessTarget: Float,
    val backgroundBrightnessTarget: Float
)

data class Tolerance(
    val positionMargin: Float,
    val areaMargin: Float
)

data class DirectorConfig(
    val targetComposition: TargetComposition,
    val targetLighting: TargetLighting,
    val tolerance: Tolerance
)

object ConfigManager {
    fun loadConfig(context: Context, filename: String = "config.json"): DirectorConfig? {
        return try {
            val inputStream = context.assets.open(filename)
            val jsonString = inputStream.bufferedReader().use(BufferedReader::readText)
            val jsonObject = JSONObject(jsonString)

            val compObj = jsonObject.getJSONObject("target_composition")
            val targetComposition = TargetComposition(
                subjectCenterX = compObj.getDouble("subject_center_x").toFloat(),
                subjectCenterY = compObj.getDouble("subject_center_y").toFloat(),
                subjectAreaRatio = compObj.getDouble("subject_area_ratio").toFloat()
            )

            val lightObj = jsonObject.getJSONObject("target_lighting")
            val targetLighting = TargetLighting(
                isBacklit = lightObj.getBoolean("is_backlit"),
                subjectBrightnessTarget = lightObj.getDouble("subject_brightness_target").toFloat(),
                backgroundBrightnessTarget = lightObj.getDouble("background_brightness_target").toFloat()
            )

            val tolObj = jsonObject.getJSONObject("tolerance")
            val tolerance = Tolerance(
                positionMargin = tolObj.getDouble("position_margin").toFloat(),
                areaMargin = tolObj.getDouble("area_margin").toFloat()
            )

            DirectorConfig(targetComposition, targetLighting, tolerance)
        } catch (e: Exception) {
            e.printStackTrace()
            null
        }
    }
}
