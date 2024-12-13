package com.capstone_bangkit.nutrise.ui.home

import android.content.Intent
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.Fragment
import com.capstone_bangkit.nutrise.databinding.FragmentHomeBinding
import com.capstone_bangkit.nutrise.ui.home.camera.CameraActivity
import com.google.firebase.Firebase
import com.google.firebase.auth.auth

class HomeFragment : Fragment() {

    private var _binding : FragmentHomeBinding? = null
    private val binding get() = _binding!!

    override fun onCreateView(
        inflater : LayoutInflater,
        container : ViewGroup?,
        savedInstanceState : Bundle?
    ) : View {
        _binding = FragmentHomeBinding.inflate(LayoutInflater.from(context), container, false)
        return binding.root
    }

    override fun onViewCreated(view : View, savedInstanceState : Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        val currentUser = Firebase.auth.currentUser
        binding.nameUser.text = currentUser?.displayName.toString()

        binding.buttonStart.setOnClickListener {
            startActivity(Intent(requireContext(), CameraActivity::class.java))
        }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}