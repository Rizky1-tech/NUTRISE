package com.capstone_bangkit.nutrise.ui.setting

import android.content.Intent
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.CompoundButton
import android.widget.Toast
import androidx.appcompat.app.AppCompatDelegate
import androidx.fragment.app.Fragment
import androidx.fragment.app.viewModels
import androidx.lifecycle.Observer
import com.capstone_bangkit.nutrise.databinding.FragmentSettingBinding
import com.capstone_bangkit.nutrise.ui.auth.login.LoginActivity
import com.capstone_bangkit.nutrise.ui.setting.history.HistoryActivity
import com.capstone_bangkit.nutrise.ui.setting.useraccount.AccountActivity
import com.capstone_bangkit.nutrise.userpref.UserPreferences
import com.capstone_bangkit.nutrise.userpref.dataStore

class SettingFragment : Fragment() {

    private var _binding : FragmentSettingBinding? = null
    private val binding get() = _binding!!

    private val settingViewModel : SettingViewModel by viewModels {
        SettingViewModelFactory(
            requireContext(),
            UserPreferences.getInstance(requireContext().applicationContext.dataStore),
            requireActivity().application
        )
    }

    override fun onCreateView(
        inflater : LayoutInflater,
        container : ViewGroup?,
        savedInstanceState : Bundle?
    ) : View {
        _binding = FragmentSettingBinding.inflate(LayoutInflater.from(context), container, false)
        return binding.root
    }

    override fun onViewCreated(view : View, savedInstanceState : Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        observeNameUser()
        observeHistoryCount()

        binding.apply {
            logoutButton.setOnClickListener {
                logout()
            }
            switchNightMode.setOnCheckedChangeListener { _ : CompoundButton?, isChecked : Boolean ->
                settingViewModel.saveThemeSetting(isChecked)
            }

            account.setOnClickListener {
                startActivity(Intent(requireContext(), AccountActivity::class.java))
            }

            history.setOnClickListener {
                startActivity(Intent(requireContext(), HistoryActivity::class.java))
            }
        }
        setTheme()
    }

    private fun observeHistoryCount() {
        settingViewModel.historyCount.observe(viewLifecycleOwner) { count ->
            binding.tvTotalHistoryValue.text = count.toString()
        }
    }

    private fun observeNameUser() {
        settingViewModel.username.observe(viewLifecycleOwner) { updateName ->
            binding.tvUsernameValue.text = updateName
        }
    }

    private fun setTheme() {
        settingViewModel.getThemeSettings()
            .observe(viewLifecycleOwner) { isDarkModeActive : Boolean ->
                if (view == null) return@observe
                if (isDarkModeActive) {
                    AppCompatDelegate.setDefaultNightMode(AppCompatDelegate.MODE_NIGHT_YES)
                    binding.switchNightMode.isChecked = true
                } else {
                    AppCompatDelegate.setDefaultNightMode(AppCompatDelegate.MODE_NIGHT_NO)
                    binding.switchNightMode.isChecked = false
                }
            }
    }

    private fun logout() {
        settingViewModel.logout()

        settingViewModel.logoutStatus.observe(viewLifecycleOwner, Observer { isSuccess ->
            if (isSuccess) {
                showToast("Logout Berhasil")
                navigateToLoginScreen()
            } else {
                showToast("Gagal")
            }
        })
    }

    private fun showToast(message : String) {
        Toast.makeText(requireContext(), message, Toast.LENGTH_SHORT).show()
    }

    private fun navigateToLoginScreen() {
        var intent = Intent(requireContext(), LoginActivity::class.java)
        intent.flags = Intent.FLAG_ACTIVITY_CLEAR_TASK or Intent.FLAG_ACTIVITY_NEW_TASK
        startActivity(intent)
        activity?.finish()
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}