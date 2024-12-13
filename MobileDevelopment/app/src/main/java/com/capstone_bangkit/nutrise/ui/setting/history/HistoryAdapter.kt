package com.capstone_bangkit.nutrise.ui.setting.history

import android.annotation.SuppressLint
import android.view.LayoutInflater
import android.view.ViewGroup
import androidx.recyclerview.widget.RecyclerView
import com.bumptech.glide.Glide
import com.capstone_bangkit.nutrise.R
import com.capstone_bangkit.nutrise.database.History
import com.capstone_bangkit.nutrise.databinding.ItemHistoryBinding

class HistoryAdapter(
    private var historyList : List<History>
) : RecyclerView.Adapter<HistoryAdapter.HistoryViewHolder>() {

    inner class HistoryViewHolder(private val binding : ItemHistoryBinding) :
        RecyclerView.ViewHolder(binding.root) {
        fun bind(history : History) {
            with(binding) {
                tvResult.text = history.result
                tvDate.text = history.date
                Glide.with(imageHistory.context)
                    .load(history.imageClassifier)
                    .placeholder(R.drawable.ic_placeholder)
                    .into(imageHistory)
            }
        }
    }

    override fun onCreateViewHolder(parent : ViewGroup, viewType : Int) : HistoryViewHolder {
        val binding = ItemHistoryBinding.inflate(
            LayoutInflater.from(parent.context),
            parent,
            false
        )
        return HistoryViewHolder(binding)
    }

    override fun onBindViewHolder(holder : HistoryViewHolder, position : Int) {
        holder.bind(historyList[position])
    }

    override fun getItemCount() : Int = historyList.size

    // Update data adapter
    @SuppressLint("NotifyDataSetChanged")
    fun updateData(newList : List<History>) {
        historyList = newList
        notifyDataSetChanged()  // Notify adapter that data has changed
    }
}

