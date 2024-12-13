package com.capstone_bangkit.nutrise.ui.setting.history

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.LiveData
import com.capstone_bangkit.nutrise.database.History
import com.capstone_bangkit.nutrise.repository.HistoryRepository

class HistoryViewModel(application: Application) : AndroidViewModel(application) {
    private val historyRepository: HistoryRepository = HistoryRepository(application)

    // LiveData untuk mengamati semua data history
    val allHistory: LiveData<List<History>> = historyRepository.getAllHistory()

    // Fungsi untuk menghapus semua riwayat
    fun deleteAllHistory() {
        historyRepository.deleteAllHistory()
    }
}