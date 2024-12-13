package com.capstone_bangkit.nutrise.helper

import androidx.recyclerview.widget.DiffUtil
import com.capstone_bangkit.nutrise.database.History

class HistoryDiffCallback : DiffUtil.ItemCallback<History>() {
    override fun areItemsTheSame(
        oldItem : History,
        newItem : History
    ) : Boolean {
        return oldItem.id == newItem.id
    }

    override fun areContentsTheSame(
        oldItem : History,
        newItem : History
    ) : Boolean {
        return oldItem == newItem
    }

}