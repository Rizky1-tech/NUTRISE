package com.capstone_bangkit.nutrise.ui

import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import com.capstone_bangkit.nutrise.ui.auth.firebase.repository.AuthRepository
import com.google.firebase.auth.FirebaseUser

class MainViewModel(private val authRepository : AuthRepository) : ViewModel() {

    private val _currentUser = MutableLiveData<FirebaseUser?>()
    val currentUser : LiveData<FirebaseUser?> get() = _currentUser

//    fun fetchCurrentUser() {
//        val user = authRepository.getCurrentUser()
//        _currentUser.value = user
//    }
}