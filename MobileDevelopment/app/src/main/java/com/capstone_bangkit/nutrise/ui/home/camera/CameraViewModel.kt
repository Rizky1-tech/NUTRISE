package com.capstone_bangkit.nutrise.ui.home.camera

import android.app.Application
import androidx.lifecycle.ViewModel
import com.capstone_bangkit.nutrise.database.History
import com.capstone_bangkit.nutrise.repository.HistoryRepository

class CameraViewModel(application : Application) : ViewModel() {

    private val mHistoryRepository : HistoryRepository = HistoryRepository(application)

    fun insert(history : History) {
        mHistoryRepository.insert(history)
    }
}