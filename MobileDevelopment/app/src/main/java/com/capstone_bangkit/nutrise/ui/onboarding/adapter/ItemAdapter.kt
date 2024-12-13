package com.capstone_bangkit.nutrise.ui.onboarding.adapter

import android.view.LayoutInflater
import android.view.ViewGroup
import androidx.recyclerview.widget.RecyclerView
import com.bumptech.glide.Glide
import com.capstone_bangkit.nutrise.ui.onboarding.model.OnboardingItem
import com.capstone_bangkit.nutrise.databinding.ItemOnboardingBinding

class ItemAdapter(private val onboardingItems: List<OnboardingItem>) :
    RecyclerView.Adapter<ItemAdapter.OnboardingViewHolder>() {

    class OnboardingViewHolder(private val binding: ItemOnboardingBinding) :
        RecyclerView.ViewHolder(binding.root) {
        fun bind(onboardingItem: OnboardingItem) {
            binding.apply {
                title1.text = onboardingItem.title
                description.text = onboardingItem.description
                Glide.with(itemView)
                    .load(onboardingItem.imageResId)
                    .into(imageView)
            }
        }
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): OnboardingViewHolder {
        val binding =
            ItemOnboardingBinding.inflate(LayoutInflater.from(parent.context), parent, false)
        return OnboardingViewHolder(binding)
    }

    override fun onBindViewHolder(holder: OnboardingViewHolder, position: Int) {
        holder.bind(onboardingItems[position])
    }

    override fun getItemCount(): Int = onboardingItems.size
}