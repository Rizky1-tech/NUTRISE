package com.capstone_bangkit.nutrise.data.retrofit.api

import com.capstone_bangkit.nutrise.data.retrofit.response.FileUploadResponse
import okhttp3.MultipartBody
import retrofit2.Call
import retrofit2.http.Multipart
import retrofit2.http.POST
import retrofit2.http.Part

interface ApiService {

    // Endpoint untuk mengirim gambar dan melakukan analisis
    @Multipart // Tambahkan anotasi @Multipart di sini
    @POST("predict")
    fun analyzeImage(
        @Part image : MultipartBody.Part // Parameter @Part untuk mengirimkan gambar
    ) : Call<FileUploadResponse>
}