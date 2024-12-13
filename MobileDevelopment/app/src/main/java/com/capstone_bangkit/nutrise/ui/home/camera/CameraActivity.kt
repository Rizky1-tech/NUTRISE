package com.capstone_bangkit.nutrise.ui.home.camera

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.ViewModelProvider
import com.capstone_bangkit.nutrise.data.retrofit.api.ApiConfig
import com.capstone_bangkit.nutrise.data.retrofit.api.ApiService
import com.capstone_bangkit.nutrise.data.retrofit.response.FileUploadResponse
import com.capstone_bangkit.nutrise.database.History
import com.capstone_bangkit.nutrise.databinding.ActivityCameraBinding
import com.capstone_bangkit.nutrise.helper.DateHelper
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.RequestBody.Companion.asRequestBody
import retrofit2.Call
import retrofit2.Callback
import retrofit2.Response
import java.io.File
import java.io.FileOutputStream
import java.io.IOException

class CameraActivity : AppCompatActivity() {

    private lateinit var binding : ActivityCameraBinding
    private var imageBitmap : Bitmap? = null
    private lateinit var cameraViewModel : CameraViewModel

    override fun onCreate(savedInstanceState : Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityCameraBinding.inflate(layoutInflater)
        setContentView(binding.root)

        cameraViewModel = ViewModelProvider(
            this,
            ViewModelFactory.getInstance(application)
        )[CameraViewModel::class.java]

        // Menangani klik untuk membuka galeri
        binding.button.setOnClickListener {
            requestGalleryPermission()
        }

        // Menangani klik untuk membuka kamera
        binding.button2.setOnClickListener {
            requestCameraPermission()
        }

        // Menangani klik untuk mengirim gambar untuk analisis
        binding.analyzeButton.setOnClickListener {
            if (imageBitmap != null) {
                analyzeImage(imageBitmap!!)
            } else {
                Toast.makeText(this, "Pilih atau ambil gambar terlebih dahulu", Toast.LENGTH_SHORT)
                    .show()
            }
        }
    }

    private fun analyzeImage(bitmap : Bitmap) {
        Toast.makeText(this, "Mengirim gambar untuk analisis...", Toast.LENGTH_SHORT).show()

        // Menyimpan gambar ke file temporer
        val file = createTempFileFromBitmap(bitmap)

        if (file != null) {
            val requestBody = file.asRequestBody("multipart/form-data".toMediaTypeOrNull())
            val filePart = MultipartBody.Part.createFormData("file", file.name, requestBody)

            // Mengirim gambar ke API untuk analisis
            val apiService : ApiService = ApiConfig.getApiService()
            apiService.analyzeImage(filePart).enqueue(object : Callback<FileUploadResponse> {
                override fun onResponse(
                    call : Call<FileUploadResponse>,
                    response : Response<FileUploadResponse>
                ) {
                    if (response.isSuccessful) {
                        val analysisResult = response.body()

                        if (analysisResult != null) {
                            // Menampilkan hasil analisis menggunakan Toast
                            val resultText = "Makanan: ${analysisResult.name}\n" +
                                    "Kalori: ${analysisResult.calories}\n" +
                                    "Karbohidrat: ${analysisResult.carbohydrates}"
                            Toast.makeText(this@CameraActivity, resultText, Toast.LENGTH_LONG).show()

                            // Menyimpan hasil analisis ke database
                            saveAnalysisResultToDatabase(analysisResult)
                        } else {
                            Toast.makeText(
                                this@CameraActivity,
                                "Gagal menganalisis gambar",
                                Toast.LENGTH_SHORT
                            ).show()
                        }

                    } else {
                        Toast.makeText(
                            this@CameraActivity,
                            "Gagal menganalisis gambar. Kode error: ${response.code()}",
                            Toast.LENGTH_SHORT
                        ).show()
                    }
                }

                override fun onFailure(call : Call<FileUploadResponse>, t : Throwable) {
                    Toast.makeText(
                        this@CameraActivity,
                        "Terjadi kesalahan: ${t.message}",
                        Toast.LENGTH_SHORT
                    ).show()
                    Log.e("CameraActivity", "Error analyzing image", t)
                }
            })
        } else {
            Toast.makeText(this, "Gagal membuat file gambar", Toast.LENGTH_SHORT).show()
        }
    }

    private fun createTempFileFromBitmap(bitmap : Bitmap) : File? {
        return try {
            // Membuat file temporer
            val file = File(cacheDir, "image_${System.currentTimeMillis()}.jpg")
            val fileOutputStream = FileOutputStream(file)
            bitmap.compress(Bitmap.CompressFormat.JPEG, 100, fileOutputStream)
            fileOutputStream.flush()
            fileOutputStream.close()
            file
        } catch (e : IOException) {
            Log.e("CameraActivity", "Error creating temp file", e)
            null
        }
    }

    private fun saveAnalysisResultToDatabase(result : FileUploadResponse) {
        val currentDate = DateHelper.getCurrentDate()
        val history = History(
            imageClassifier = result.name ?: "Tidak Dikenali",
            result = "Kalori: ${result.calories}, Karbohidrat: ${result.carbohydrates}",
            date = currentDate
        )
        cameraViewModel.insert(history)
    }

    private fun requestGalleryPermission() {
        val permission =
            if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.TIRAMISU) {
                Manifest.permission.READ_MEDIA_IMAGES
            } else {
                Manifest.permission.READ_EXTERNAL_STORAGE
            }

        if (ContextCompat.checkSelfPermission(
                this,
                permission
            ) == android.content.pm.PackageManager.PERMISSION_GRANTED
        ) {
            openGalleryLauncher.launch("image/*")
        } else {
            ActivityCompat.requestPermissions(this, arrayOf(permission), REQUEST_CODE_GALLERY)
        }
    }

    private val openGalleryLauncher =
        registerForActivityResult(ActivityResultContracts.GetContent()) { uri : Uri? ->
            uri?.let {
                try {
                    val bitmap = MediaStore.Images.Media.getBitmap(contentResolver, it)
                    binding.previewImageView.setImageBitmap(bitmap)
                    imageBitmap = bitmap
                } catch (e : Exception) {
                    Log.e("CameraActivity", "Error loading image from gallery", e)
                    Toast.makeText(this, "Gagal memuat gambar dari galeri", Toast.LENGTH_SHORT)
                        .show()
                }
            }
        }

    private val openCameraLauncher =
        registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
            if (result.resultCode == Activity.RESULT_OK) {
                val bitmap = result.data?.extras?.get("data") as? Bitmap
                if (bitmap != null) {
                    binding.previewImageView.setImageBitmap(bitmap)
                    imageBitmap = bitmap
                } else {
                    Toast.makeText(this, "Gagal menangkap gambar", Toast.LENGTH_SHORT).show()
                }
            }
        }

    private fun requestCameraPermission() {
        if (ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.CAMERA
            ) == android.content.pm.PackageManager.PERMISSION_GRANTED
        ) {
            launchCamera()
        } else {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.CAMERA),
                REQUEST_CODE_CAMERA
            )
        }
    }

    private fun launchCamera() {
        val cameraIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
        if (cameraIntent.resolveActivity(packageManager) != null) {
            openCameraLauncher.launch(cameraIntent)
        } else {
            Toast.makeText(this, "Tidak ada aplikasi kamera yang tersedia", Toast.LENGTH_SHORT)
                .show()
        }
    }

    companion object {
        private const val REQUEST_CODE_GALLERY = 100
        private const val REQUEST_CODE_CAMERA = 101
    }
}
