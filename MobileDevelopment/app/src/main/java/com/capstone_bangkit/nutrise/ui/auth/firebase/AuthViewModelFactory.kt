package com.capstone_bangkit.nutrise.ui.auth.firebase

import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import com.capstone_bangkit.nutrise.ui.MainViewModel
import com.capstone_bangkit.nutrise.ui.auth.firebase.repository.AuthRepository
import com.capstone_bangkit.nutrise.ui.auth.login.LoginViewModel
import com.capstone_bangkit.nutrise.ui.auth.register.RegisterViewModel
import com.capstone_bangkit.nutrise.ui.setting.useraccount.AccountViewModel

class AuthViewModelFactory(private val authRepository : AuthRepository) :
    ViewModelProvider.Factory {
    @Suppress("UNCHECKED_CAST")
    override fun <T : ViewModel> create(modelClass : Class<T>) : T {
        return if (modelClass.isAssignableFrom(LoginViewModel::class.java)) {
            LoginViewModel(authRepository) as T
        } else if (modelClass.isAssignableFrom(MainViewModel::class.java)) {
            MainViewModel(authRepository) as T
        } else if (modelClass.isAssignableFrom(RegisterViewModel::class.java)) {
            RegisterViewModel(authRepository) as T
        } else if (modelClass.isAssignableFrom(AccountViewModel::class.java)) {
            AccountViewModel(authRepository) as T
        } else {
            throw IllegalArgumentException("Unknown ViewModel class")
        }
    }
}