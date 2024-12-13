package com.capstone_bangkit.nutrise.ui.news

import android.os.Parcelable
import kotlinx.parcelize.Parcelize

@Parcelize
data class News(
    val name: String,
    val description: String,
    val photo: Int
) : Parcelable