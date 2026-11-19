# 📱 PANDUAN INTEGRASI ONNX DI ANDROID (KOTLIN + JETPACK COMPOSE)

Dokumen ini memuat panduan langkah-demi-langkah mengintegrasikan model machine learning JagaMata ke dalam aplikasi Android menggunakan **ONNX Runtime Android** (`com.microsoft.onnxruntime`).

---

## 1. Spesifikasi Model Ter-Export

| Parameter | Model Task 1 (Head Pitch Regressor) | Model Task 2 (Myopia Classifier) |
|---|---|---|
| **Arsitektur Backbone** | MobileNetV3-Small (Multi-Task 3D Pose) | EfficientNet-B0 |
| **File ONNX FP32** | `jagamata_pitch_tuned.onnx` (5.81 MB) | `jagamata_fundus_classifier.onnx` (15.27 MB) |
| **File ONNX INT8 (Quantized)** | `jagamata_pitch_tuned_int8.onnx` (1.62 MB) | - |
| **Dimensi Input Tensor** | `[1, 3, 224, 224]` (Float32) | `[1, 3, 224, 224]` (Float32) |
| **Normalisasi Citra** | Mean `[0.485, 0.456, 0.406]`, Std `[0.229, 0.224, 0.225]` | Mean `[0.485, 0.456, 0.406]`, Std `[0.229, 0.224, 0.225]` |
| **Dimensi Output** | `[1, 3]` -> `[Pitch, Yaw, Roll]` (° derajat) | `[1, 2]` -> Logits `[Normal, Myopia]` |
| **Target Latensi Mobile** | **~12 ms** (NNAPI / GPU Delegate) | **~35 ms** |

---

## 2. Setup Dependency Android (`app/build.gradle.kts`)

Tambahkan library resmi ONNX Runtime Android dan CameraX:

```kotlin
dependencies {
    // ONNX Runtime Android (Mendukung NNAPI, CPU, dan GPU acceleration)
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.17.0")

    // CameraX untuk tangkapan frame kamera real-time
    val cameraxVersion = "1.3.1"
    implementation("androidx.camera:camera-core:$cameraxVersion")
    implementation("androidx.camera:camera-camera2:$cameraxVersion")
    implementation("androidx.camera:camera-lifecycle:$cameraxVersion")
    implementation("androidx.camera:camera-view:$cameraxVersion")
}
```

Salin file model `.onnx` ke folder:
`app/src/main/assets/jagamata_pitch_tuned_int8.onnx`

---

## 3. Implementasi Kotlin HeadPoseEstimator.kt

```kotlin
package com.jagamata.ai.pose

import android.content.Context
import android.graphics.Bitmap
import ai.onnxruntime.*
import java.nio.FloatBuffer
import kotlin.math.abs

data class HeadPoseResult(
    val pitchDeg: Float,
    val yawDeg: Float,
    val rollDeg: Float,
    val isFacingScreen: Boolean, // Guardrail |yaw| < 45°
    val isPostureGood: Boolean   // Evaluasi kemiringan kepala anak
)

class HeadPoseEstimator(private val context: Context) {
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    init {
        val modelBytes = context.assets.open("jagamata_pitch_tuned_int8.onnx").readBytes()
        val opts = OrtSession.SessionOptions().apply {
            // Aktifkan NNAPI accelerator jika perangkat mendukung hardware NPU
            try {
                addNnapi()
            } catch (e: Exception) {
                setInterOpNumThreads(2)
                setIntraOpNumThreads(4)
            }
        }
        session = env.createSession(modelBytes, opts)
    }

    fun estimate(faceBitmap: Bitmap): HeadPoseResult {
        // 1. Resize ke 224x224
        val resized = Bitmap.createScaledBitmap(faceBitmap, 224, 224, true)
        
        // 2. Normalisasi ImageNet (Mean & Std) ke NCHW FloatBuffer
        val inputBuffer = FloatBuffer.allocate(1 * 3 * 224 * 224)
        val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
        val std = floatArrayOf(0.229f, 0.224f, 0.225f)
        
        val pixels = IntArray(224 * 224)
        resized.getPixels(pixels, 0, 224, 0, 0, 224, 224)
        
        // Channel R, G, B plane terpisah (NCHW)
        for (c in 0 until 3) {
            for (i in pixels.indices) {
                val pixel = pixels[i]
                val channelVal = when (c) {
                    0 -> ((pixel shr 16) and 0xFF) / 255.0f
                    1 -> ((pixel shr 8) and 0xFF) / 255.0f
                    else -> (pixel and 0xFF) / 255.0f
                }
                val normalized = (channelVal - mean[c]) / std[c]
                inputBuffer.put(normalized)
            }
        }
        inputBuffer.flip()

        // 3. Jalankan Inferensi ONNX
        val inputTensor = OnnxTensor.createTensor(env, inputBuffer, longArrayOf(1, 3, 224, 224))
        val outputs = session.run(mapOf("input_face_image" to inputTensor))
        
        @Suppress("UNCHECKED_CAST")
        val rawResults = (outputs[0].value as Array<FloatArray>)[0]
        
        val pitch = rawResults[0]
        val yaw = rawResults[1]
        val roll = rawResults[2]

        // 4. Guardrail & Logika Klinis JagaMata
        // Jika |yaw| >= 45°, anak sedang memalingkan kepala dari layar
        val isFacing = abs(yaw) < 45.0f
        
        // Postur aman menatap gadget: Pitch berada dalam rentang -15° s.d. +10°
        val isGood = isFacing && (pitch >= -15.0f && pitch <= 10.0f)

        inputTensor.close()
        outputs.close()

        return HeadPoseResult(
            pitchDeg = pitch,
            yawDeg = yaw,
            rollDeg = roll,
            isFacingScreen = isFacing,
            isPostureGood = isGood
        )
    }

    fun close() {
        session.close()
        env.close()
    }
}
```

---

## 4. Integrasi UI Jetpack Compose

```kotlin
@Composable
fun PostureFeedbackBadge(pose: HeadPoseResult) {
    val (color, text) = when {
        !pose.isFacingScreen -> Color.Yellow to "⚠️ Harap Menghadap Layar"
        pose.isPostureGood -> Color.Green to "✅ Postur Kepala Baik (Pitch: ${pose.pitchDeg.toInt()}°)"
        else -> Color.Red to "🔴 Kepala Terlalu Menunduk (${pose.pitchDeg.toInt()}°)"
    }

    Surface(
        color = color.copy(alpha = 0.2f),
        shape = RoundedCornerShape(12.dp),
        modifier = Modifier.padding(16.dp)
    ) {
        Text(
            text = text,
            color = color,
            style = MaterialTheme.typography.titleMedium,
            modifier = Modifier.padding(horizontal = 16.dp, vertical = 8.dp)
        )
    }
}
```