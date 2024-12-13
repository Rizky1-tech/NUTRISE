package com.capstone_bangkit.nutrise.ui.auth.register

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.capstone_bangkit.nutrise.ui.auth.firebase.model.UserModelFirebase
import com.capstone_bangkit.nutrise.ui.auth.firebase.repository.AuthRepository
import com.capstone_bangkit.nutrise.ui.auth.firebase.repository.Result
import kotlinx.coroutines.launch

class RegisterViewModel(private val authRepository : AuthRepository) : ViewModel() {

    fun registerUser(
        name : String,
        email : String,
        password : String,
        callback : (Result<UserModelFirebase>) -> Unit
    ) {
        viewModelScope.launch {
            val result = authRepository.register(name, email, password)
            callback(result)
        }
    }
}