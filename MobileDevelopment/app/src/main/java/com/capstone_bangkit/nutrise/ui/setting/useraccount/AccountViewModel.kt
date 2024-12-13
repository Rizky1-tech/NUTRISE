package com.capstone_bangkit.nutrise.ui.setting.useraccount

import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.capstone_bangkit.nutrise.ui.auth.firebase.model.UserModelFirebase
import com.capstone_bangkit.nutrise.ui.auth.firebase.repository.AuthRepository
import com.capstone_bangkit.nutrise.ui.auth.firebase.repository.Result
import kotlinx.coroutines.launch

class AccountViewModel(private val authRepository : AuthRepository) : ViewModel() {
    private val _updateUserResult = MutableLiveData<Result<UserModelFirebase>>()
    val updateUserResult : LiveData<Result<UserModelFirebase>> get() = _updateUserResult

    private val _userDataResult = MutableLiveData<Result<UserModelFirebase>>()
    val userDataResult : LiveData<Result<UserModelFirebase>> get() = _userDataResult

    init {
        // Ambil data pengguna saat halaman akun pertama kali dibuka
        getUserData()
    }

    // Fungsi untuk mendapatkan data pengguna
    fun getUserData() {
        _userDataResult.value = Result.Loading
        viewModelScope.launch {
            val result = authRepository.getUser()
            _userDataResult.value = result
        }
    }

    // Perbarui data pengguna, termasuk nama, email, password (dan password baru jika ada)
    fun updateUser(name: String? = null, email: String? = null, password: String? = null, newPassword: String? = null) {
        _updateUserResult.value = Result.Loading
        viewModelScope.launch {
            val result = authRepository.updateUser(name, email, password, newPassword)
            _updateUserResult.value = result
        }
    }
}