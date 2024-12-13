package com.capstone_bangkit.nutrise.database

import android.os.Parcelable
import androidx.room.ColumnInfo
import androidx.room.Entity
import androidx.room.PrimaryKey
import kotlinx.parcelize.Parcelize

@Entity
@Parcelize
data class History(
    @PrimaryKey(autoGenerate = true)
    var id : Int = 0,

    @ColumnInfo(name = "image")
    var imageClassifier : String? = null,

    @ColumnInfo(name = "result")
    var result : String? = null,

    @ColumnInfo(name = "date")
    var date : String? = null
) : Parcelable