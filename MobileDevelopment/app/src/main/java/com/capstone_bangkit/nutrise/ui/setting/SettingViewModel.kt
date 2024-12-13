package com.capstone_bangkit.nutrise.ui.setting

import android.annotation.SuppressLint
import android.app.Application
import android.content.Context
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import androidx.lifecycle.asLiveData
import androidx.lifecycle.viewModelScope
import com.capstone_bangkit.nutrise.repository.HistoryRepository
import com.capstone_bangkit.nutrise.ui.auth.firebase.repository.AuthRepository
import com.capstone_bangkit.nutrise.userpref.UserPreferences
import com.google.firebase.Firebase
import com.google.firebase.auth.auth
import kotlinx.coroutines.launch

@SuppressLint("StaticFieldLeak")
class SettingViewModel(
    application: Application,
    private val context: Context,
    private val pref: UserPreferences
) : ViewModel() {
    private val authRepository = AuthRepository()
    private val historyRepository = HistoryRepository(application)

    val logoutStatus = MutableLiveData<Boolean>()

    private val _username = MutableLiveData<String>()
    val username: LiveData<String> get() = _username

    val historyCount: LiveData<Int> = historyRepository.getHistoryCount()

    init {
        // Mengambil data pengguna dari Firebase
        loadUserData()
    }

    private fun loadUserData() {
        val currentUser = Firebase.auth.currentUser
        _username.value = currentUser?.displayName ?: "User belum login"
    }

    fun getThemeSettings(): LiveData<Boolean> {
        return pref.getThemeSetting().asLiveData()
    }

    fun saveThemeSetting(isDarkModeActive: Boolean) {
        viewModelScope.launch {
            pref.saveThemeSetting(isDarkModeActive)
        }
    }

    fun logout() {
        authRepository.logoutUser(context, onSuccess = {
            logoutStatus.value = true
        }, onFailure = {
            logoutStatus.value = false
        })
    }
}
