package com.capstone_bangkit.nutrise.apps

import android.app.Application
import androidx.appcompat.app.AppCompatDelegate
import com.capstone_bangkit.nutrise.userpref.UserPreferences
import com.capstone_bangkit.nutrise.userpref.dataStore
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

class Apps : Application() {
    override fun onCreate() {
        super.onCreate()
        val userPreferences = UserPreferences.getInstance(applicationContext.dataStore)

        CoroutineScope(Dispatchers.Main).launch {
            userPreferences.getThemeSetting().collect {isDarkModeActive ->
                if (isDarkModeActive) {
                    AppCompatDelegate.setDefaultNightMode(AppCompatDelegate.MODE_NIGHT_YES)
                } else {
                    AppCompatDelegate.setDefaultNightMode(AppCompatDelegate.MODE_NIGHT_NO)
                }
            }
        }
    }
}