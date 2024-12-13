package com.capstone_bangkit.nutrise.ui.setting.useraccount

import android.os.Build
import android.os.Bundle
import android.view.MenuItem
import android.view.View
import android.view.WindowInsets
import android.view.WindowManager
import android.widget.Toast
import androidx.activity.viewModels
import androidx.appcompat.app.AppCompatActivity
import androidx.core.widget.doOnTextChanged
import com.capstone_bangkit.nutrise.databinding.ActivityAccountBinding
import com.capstone_bangkit.nutrise.ui.auth.firebase.repository.AuthRepository
import com.capstone_bangkit.nutrise.ui.auth.firebase.repository.Result
import com.capstone_bangkit.nutrise.ui.auth.firebase.AuthViewModelFactory

class AccountActivity : AppCompatActivity() {

    private lateinit var binding: ActivityAccountBinding
    private val accountViewModel: AccountViewModel by viewModels {
        AuthViewModelFactory(AuthRepository())
    }

    private var initialName: String? = null
    private var initialEmail: String? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityAccountBinding.inflate(layoutInflater)
        setContentView(binding.root)
        setupView()

        setSupportActionBar(binding.toolbar)
        supportActionBar?.setDisplayHomeAsUpEnabled(true)
        supportActionBar?.setDisplayShowHomeEnabled(true)

        observeUserData()
        setSaveButtonState()
        setupSaveButton()
    }

    private fun setupView() {
        @Suppress("DEPRECATION")
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            window.insetsController?.hide(WindowInsets.Type.statusBars())
        } else {
            window.setFlags(
                WindowManager.LayoutParams.FLAG_FULLSCREEN,
                WindowManager.LayoutParams.FLAG_FULLSCREEN
            )
        }
    }

    private fun observeUserData() {
        accountViewModel.userDataResult.observe(this) { result ->
            when (result) {
                is Result.Loading -> {
                    // Optional: Show loading indicator here
                    binding.loading.visibility = View.VISIBLE
                }
                is Result.Success -> {
                    binding.loading.visibility = View.GONE
                    val user = result.data
                    initialName = user.name
                    initialEmail = user.email
                    // Tampilkan nama dan email pengguna
                    binding.usernameInput.setText(user.name)
                    binding.emailInput.setText(user.email)
                }
                is Result.Error -> {
                    binding.loading.visibility = View.GONE

                    Toast.makeText(this, "Gagal memuat data: ${result.error}", Toast.LENGTH_SHORT).show()
                }
            }
        }
        // Ambil data pengguna saat halaman pertama kali dibuka
        accountViewModel.getUserData()
    }

    private fun setSaveButtonState() {
        val inputs = listOf(binding.usernameInput, binding.emailInput, binding.passwordInput, binding.newPasswordInput)
        for (input in inputs) {
            input.doOnTextChanged { _, _, _, _ -> validateInputs() }
        }
    }

    private fun validateInputs() {
        val nameChanged = binding.usernameInput.text.toString() != initialName
        val emailChanged = binding.emailInput.text.toString() != initialEmail
        val passwordEntered = binding.passwordInput.text.toString().isNotEmpty()
        val newPasswordEntered = binding.newPasswordInput.text.toString().isNotEmpty()

        // Enable save button if there's any change or password is entered
        binding.saveButton.isEnabled = nameChanged || emailChanged || passwordEntered || newPasswordEntered
    }

    private fun setupSaveButton() {
        binding.saveButton.setOnClickListener {
            val name = binding.usernameInput.text.toString().takeIf { it != initialName }
            val email = binding.emailInput.text.toString().takeIf { it != initialEmail }
            val password = binding.passwordInput.text.toString().takeIf { it.isNotEmpty() }
            val newPassword = binding.newPasswordInput.text.toString().takeIf { it.isNotEmpty() }

            // Cek apakah ada perubahan data
            if (password != null && newPassword != null) {
                // Update password jika password lama dan baru diisi
                accountViewModel.updateUser(name, email, password = password, newPassword = newPassword)
                observeUpdateResult()
            } else {
                // Update hanya nama dan email
                accountViewModel.updateUser(name, email)
                observeUpdateResult()
            }
        }
    }

    private fun observeUpdateResult() {
        accountViewModel.updateUserResult.observe(this) { result ->
            when (result) {
                is Result.Loading -> {
                    // Tampilkan loading indicator saat menyimpan data
                    binding.loading.visibility = View.VISIBLE
                }
                is Result.Success -> {
                    binding.loading.visibility = View.GONE
                    // Tampilkan pesan sukses setelah berhasil memperbarui data
                    Toast.makeText(this, "Data berhasil diperbarui!", Toast.LENGTH_SHORT).show()
                }
                is Result.Error -> {
                    binding.loading.visibility = View.GONE
                    // Tampilkan pesan kesalahan jika gagal memperbarui data
                    Toast.makeText(this, "Gagal memperbarui data: ${result.error}", Toast.LENGTH_SHORT).show()
                }
            }
        }
    }

    @Suppress("DEPRECATION")
    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        return when (item.itemId) {
            android.R.id.home -> {
                onBackPressed()
                true
            }
            else -> super.onOptionsItemSelected(item)
        }
    }
}
