package com.capstone_bangkit.nutrise.ui.auth.login

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.capstone_bangkit.nutrise.ui.auth.firebase.model.UserModelFirebase
import com.capstone_bangkit.nutrise.ui.auth.firebase.repository.AuthRepository
import com.capstone_bangkit.nutrise.ui.auth.firebase.repository.Result
import kotlinx.coroutines.launch

class LoginViewModel(private val authRepository : AuthRepository) : ViewModel() {
    fun login(email : String, password : String, onResult : (Result<UserModelFirebase>) -> Unit) {
        viewModelScope.launch {
            onResult(Result.Loading)
            val result = authRepository.login(email, password)
            onResult(result)
        }
    }
}