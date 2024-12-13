package com.capstone_bangkit.nutrise.ui.news

import android.view.LayoutInflater
import android.view.ViewGroup
import androidx.recyclerview.widget.RecyclerView
import com.capstone_bangkit.nutrise.databinding.ItemNewsBinding

class NewsAdapter(private val newsList: List<News>) : RecyclerView.Adapter<NewsAdapter.NewsViewHolder>() {

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): NewsViewHolder {
        val binding = ItemNewsBinding.inflate(LayoutInflater.from(parent.context), parent, false)
        return NewsViewHolder(binding)
    }

    override fun onBindViewHolder(holder: NewsViewHolder, position: Int) {
        val currentItem = newsList[position]
        holder.bind(currentItem)
    }

    override fun getItemCount(): Int = newsList.size

    inner class NewsViewHolder(private val binding: ItemNewsBinding) : RecyclerView.ViewHolder(binding.root) {

        fun bind(news: News) {
            binding.tvItemName.text = news.name
            binding.tvItemDescription.text = news.description
            binding.imgItemPhoto.setImageResource(news.photo)
        }
    }
}