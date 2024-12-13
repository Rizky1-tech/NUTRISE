package com.capstone_bangkit.nutrise.data.retrofit.response

import com.google.gson.annotations.SerializedName

data class FileUploadResponse(

    @field:SerializedName("Carbohydrates")
    val carbohydrates: Int? = null,

    @field:SerializedName("Fat")
    val fat: Any? = null,

    @field:SerializedName("Calories")
    val calories: Int? = null,

    @field:SerializedName("Name")
    val name: String? = null
)