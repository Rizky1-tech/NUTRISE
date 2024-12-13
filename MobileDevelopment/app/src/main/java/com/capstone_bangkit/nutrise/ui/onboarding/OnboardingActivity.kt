package com.capstone_bangkit.nutrise.ui.onboarding

import android.content.Intent
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.WindowInsets
import android.view.WindowManager
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.viewpager2.widget.ViewPager2
import com.capstone_bangkit.nutrise.R
import com.capstone_bangkit.nutrise.databinding.ActivityOnboardingBinding
import com.capstone_bangkit.nutrise.ui.auth.login.LoginActivity
import com.capstone_bangkit.nutrise.ui.onboarding.adapter.ItemAdapter
import com.capstone_bangkit.nutrise.ui.onboarding.model.OnboardingItem

class OnboardingActivity : AppCompatActivity() {
    private lateinit var binding : ActivityOnboardingBinding
    private lateinit var viewPager : ViewPager2
    private lateinit var adapter : ItemAdapter
    private val handler = Handler(Looper.getMainLooper())
    private lateinit var runnable : Runnable

    override fun onCreate(savedInstanceState : Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityOnboardingBinding.inflate(layoutInflater)
        setContentView(binding.root)
        setupView()

        val onboardingItems = listOf(
            OnboardingItem(
                R.drawable.onboarding1,
                getString(R.string.title_onboarding1),
                getString(R.string.description_onboarding1)
            ),
            OnboardingItem(
                R.drawable.onboarding2,
                getString(R.string.title_onboarding2),
                getString(R.string.description_onboarding2)
            ),
            OnboardingItem(
                R.drawable.onboarding3,
                getString(R.string.title_onboarding3),
                getString(R.string.description_onboarding3)
            )
        )

        // Menghubungkan adapter ke ViewPager2
        adapter = ItemAdapter(onboardingItems)
        viewPager = binding.vpImage
        viewPager.adapter = adapter

        setupAutoSlide()

        binding.button.setOnClickListener{
            startActivity(Intent(this@OnboardingActivity, LoginActivity::class.java))
        }
    }

    private fun setupAutoSlide() {
        runnable = Runnable {
            val currentItem = viewPager.currentItem
            val nextItem = if (currentItem == adapter.itemCount - 1) 0 else currentItem + 1
            viewPager.setCurrentItem(nextItem, true)
            handler.postDelayed(runnable, 3000) //jeda 5 detik
        }

        handler.postDelayed(runnable, 3000)
    }

    private fun setupView() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            window.insetsController?.hide(WindowInsets.Type.statusBars())
        } else {
            @Suppress("DEPRECATION")
            window.setFlags(
                WindowManager.LayoutParams.FLAG_FULLSCREEN,
                WindowManager.LayoutParams.FLAG_FULLSCREEN
            )
        }
        supportActionBar?.hide()
    }
}