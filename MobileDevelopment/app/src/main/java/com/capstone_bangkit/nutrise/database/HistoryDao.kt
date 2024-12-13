package com.capstone_bangkit.nutrise.database

import androidx.lifecycle.LiveData
import androidx.room.Dao
import androidx.room.Delete
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query

@Dao
interface HistoryDao {
    @Query("SELECT * FROM history")
    fun getAllHistory() : LiveData<List<History>>

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    fun insertNewHistory(history : History)

    @Delete
    fun delete(history : History)

    @Query("SELECT COUNT(*) FROM history")
    fun getHistoryCount(): LiveData<Int>

    @Query("DELETE FROM history")
    fun deleteAllHistory()
}