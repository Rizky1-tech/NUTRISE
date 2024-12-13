package com.capstone_bangkit.nutrise.repository

import android.app.Application
import androidx.lifecycle.LiveData
import com.capstone_bangkit.nutrise.database.History
import com.capstone_bangkit.nutrise.database.HistoryDao
import com.capstone_bangkit.nutrise.database.HistoryDatabase
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class HistoryRepository(application : Application) {
    private val mHistoryDao : HistoryDao
    private val executorService : ExecutorService = Executors.newSingleThreadExecutor()

    init {
        val db = HistoryDatabase.getDatabase(application)
        mHistoryDao = db.historyDao()
    }

    fun getAllHistory(): LiveData<List<History>> = mHistoryDao.getAllHistory()

    fun insert(history: History) {
        executorService.execute { mHistoryDao.insertNewHistory(history) }
    }

    fun deleteAllHistory() {
        executorService.execute { mHistoryDao.deleteAllHistory() }
    }

    fun getHistoryCount(): LiveData<Int> = mHistoryDao.getHistoryCount()
}