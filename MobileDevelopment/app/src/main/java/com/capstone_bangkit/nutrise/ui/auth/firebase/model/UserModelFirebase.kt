package com.capstone_bangkit.nutrise.ui.auth.firebase.model

import android.os.Parcelable
import kotlinx.parcelize.Parcelize

@Parcelize
data class UserModelFirebase(
    var name : String? = null,
    var email : String? = null,
    var password : String? = null
) : Parcelable