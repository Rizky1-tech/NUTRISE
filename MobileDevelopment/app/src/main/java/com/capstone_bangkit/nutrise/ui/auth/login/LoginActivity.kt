package com.capstone_bangkit.nutrise.ui.auth.login

import android.content.Intent
import android.os.Build
import android.os.Bundle
import android.text.Editable
import android.text.TextWatcher
import android.util.Patterns
import android.view.View
import android.view.WindowInsets
import android.view.WindowManager
import android.widget.Toast
import androidx.activity.viewModels
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.capstone_bangkit.nutrise.R
import com.capstone_bangkit.nutrise.databinding.ActivityLoginBinding
import com.capstone_bangkit.nutrise.ui.MainActivity
import com.capstone_bangkit.nutrise.ui.auth.firebase.AuthViewModelFactory
import com.capstone_bangkit.nutrise.ui.auth.firebase.model.UserModelFirebase
import com.capstone_bangkit.nutrise.ui.auth.firebase.repository.AuthRepository
import com.capstone_bangkit.nutrise.ui.auth.firebase.repository.Result
import com.capstone_bangkit.nutrise.ui.auth.register.RegisterActivity
import com.capstone_bangkit.nutrise.userpref.UserPreferences
import com.capstone_bangkit.nutrise.userpref.dataStore
import com.google.android.material.textfield.TextInputEditText
import com.google.android.material.textfield.TextInputLayout
import kotlinx.coroutines.launch

class LoginActivity : AppCompatActivity() {
    private lateinit var binding : ActivityLoginBinding

    private val loginViewModel : LoginViewModel by viewModels {
        AuthViewModelFactory(AuthRepository())
    }

    private lateinit var textInputLayoutEmail : TextInputLayout
    private lateinit var emailEditText : TextInputEditText

    private lateinit var textInputLayoutPassword : TextInputLayout
    private lateinit var passwordEditText : TextInputEditText

    override fun onCreate(savedInstanceState : Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityLoginBinding.inflate(layoutInflater)
        setContentView(binding.root)
        setupView()

        textInputLayoutEmail = findViewById(R.id.textInputLayoutEmail)
        emailEditText = findViewById(R.id.emailEditText)
        textInputLayoutPassword = findViewById(R.id.textInputLayoutPassword)
        passwordEditText = findViewById(R.id.passwordEditText)
        setupEmailValidation()
        setupPasswordValidation()

        binding.buttonLogin.setOnClickListener {
            val email = binding.emailEditText.text.toString()
            val password = binding.passwordEditText.text.toString()

            if (email.isNotEmpty() && password.isNotEmpty()) {
                if (password.length >= 8) {
                    loginUser(email, password)
                } else {
                    Toast.makeText(this, "Password kurang dari 8 karakter!", Toast.LENGTH_SHORT)
                        .show()
                }
            } else {
                Toast.makeText(this, "Email/Password tidak boleh kosong!", Toast.LENGTH_SHORT)
                    .show()
            }
        }

        binding.buttonRegister.setOnClickListener {
            startActivity(Intent(this, RegisterActivity::class.java))
        }
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
        supportActionBar?.hide()
    }

    private fun setupEmailValidation() {
        emailEditText.addTextChangedListener(object : TextWatcher {
            override fun beforeTextChanged(
                charSequence : CharSequence?,
                start : Int,
                count : Int,
                after : Int
            ) {
            }

            override fun onTextChanged(
                charSequence : CharSequence?,
                start : Int,
                before : Int,
                count : Int
            ) {
            }

            override fun afterTextChanged(editable : Editable?) {
                val email = editable.toString()

                if (!Patterns.EMAIL_ADDRESS.matcher(email).matches()) {
                    textInputLayoutEmail.error = "Format email tidak valid"
                } else {
                    textInputLayoutEmail.error = null
                }
            }
        })
    }

    private fun setupPasswordValidation() {
        passwordEditText.addTextChangedListener(object : TextWatcher {
            override fun beforeTextChanged(
                p0 : CharSequence?,
                p1 : Int,
                p2 : Int,
                p3 : Int
            ) {

            }

            override fun onTextChanged(
                p0 : CharSequence?,
                p1 : Int,
                p2 : Int,
                p3 : Int
            ) {

            }

            override fun afterTextChanged(p0 : Editable?) {
                val password = p0.toString()

                when {
                    password.length < 8 -> {
                        textInputLayoutPassword.error = "Password harus minimal 8 karakter"
                    }

                    else -> {
                        textInputLayoutPassword.error = null
                    }
                }
            }
        })
    }


    private fun loginUser(email : String, password : String) {
        loginViewModel.login(email, password) { result ->
            when (result) {
                is Result.Loading -> {
                    binding.loading.visibility = View.VISIBLE
                    binding.apply {
                        buttonLogin.isEnabled = false
                        buttonRegister.isEnabled = false
                    }
                }

                is Result.Success -> {
                    binding.loading.visibility = View.GONE
                    saveSession(result.data)
                    val intent = Intent(this, MainActivity::class.java)
                    intent.flags = Intent.FLAG_ACTIVITY_CLEAR_TASK or Intent.FLAG_ACTIVITY_NEW_TASK
                    startActivity(intent)
                }

                is Result.Error -> {
                    binding.loading.visibility = View.GONE
                    binding.apply {
                        buttonLogin.isEnabled = true
                        buttonRegister.isEnabled = true
                    }
                    Toast.makeText(this, "Email/Password tidak dikenal", Toast.LENGTH_SHORT).show()
                }
            }
        }
    }

    private fun saveSession(user : UserModelFirebase) {
        val userPreferences = UserPreferences.getInstance(applicationContext.dataStore)
        lifecycleScope.launch {
            userPreferences.saveSession(user)
        }
    }
}
