package com.capstone_bangkit.nutrise.ui.auth.firebase.repository

import android.content.Context
import com.capstone_bangkit.nutrise.ui.auth.firebase.model.UserModelFirebase
import com.capstone_bangkit.nutrise.userpref.UserPreferences
import com.capstone_bangkit.nutrise.userpref.dataStore
import com.google.firebase.auth.EmailAuthProvider
import com.google.firebase.auth.FirebaseAuth
import com.google.firebase.auth.FirebaseAuthException
import com.google.firebase.auth.FirebaseAuthInvalidCredentialsException
import com.google.firebase.auth.FirebaseAuthInvalidUserException
import com.google.firebase.auth.UserProfileChangeRequest
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.tasks.await

class AuthRepository {
    private val auth : FirebaseAuth = FirebaseAuth.getInstance()

    suspend fun login(email : String, password : String) : Result<UserModelFirebase> {
        return try {
            val response = auth.signInWithEmailAndPassword(email, password).await()
            val firebaseUser = response.user
            if (firebaseUser != null) {
                val user = UserModelFirebase(
                    name = firebaseUser.displayName ?: "",
                    email = firebaseUser.email ?: "",
                    password = ""
                )
                Result.Success(user)
            } else {
                Result.Error("Login failed: User not found")
            }
        } catch (e : Exception) {
            Result.Error(e.message ?: "Login error")
        }
    }

    suspend fun register(
        name : String,
        email : String,
        password : String
    ) : Result<UserModelFirebase> {
        return try {
            val response = auth.createUserWithEmailAndPassword(email, password).await()
            val user = response.user

            if (user != null) {
                val profileUpdates = UserProfileChangeRequest.Builder()
                    .setDisplayName(name)
                    .build()
                user.updateProfile(profileUpdates).await()

                val userModel = UserModelFirebase(
                    name = user.displayName ?: "",
                    email = user.email ?: "",
                    password = ""
                )

                Result.Success(userModel)
            } else {
                Result.Error("Registration failed")
            }
        } catch (e : Exception) {
            Result.Error(e.message ?: "Registration Error")
        }
    }

    @Suppress("DEPRECATION")
    suspend fun updateUser(
        name: String? = null,
        email: String? = null,
        password: String? = null,
        newPassword: String? = null
    ): Result<UserModelFirebase> {
        return try {
            val currentUser = auth.currentUser
            if (currentUser == null) {
                return Result.Error("User not logged in")
            }

            // Verifikasi ulang jika password lama disediakan
            if (password != null) {
                val credential = EmailAuthProvider.getCredential(currentUser.email!!, password)
                currentUser.reauthenticate(credential).await()  // Verifikasi ulang untuk memastikan kredensial valid
            }

            // Update display name
            name?.let {
                val profileUpdates = UserProfileChangeRequest.Builder()
                    .setDisplayName(it)
                    .build()
                currentUser.updateProfile(profileUpdates).await()
            }

            // Update email
            email?.let {
                currentUser.updateEmail(it).await()
            }

            // Update password jika newPassword disediakan
            newPassword?.let {
                currentUser.updatePassword(it).await()
            }

            // Fetch updated user information
            val updatedUser = UserModelFirebase(
                name = currentUser.displayName ?: "",
                email = currentUser.email ?: "",
                password = "" // Password tidak dikembalikan untuk alasan keamanan
            )
            Result.Success(updatedUser)
        } catch (e: FirebaseAuthInvalidCredentialsException) {
            Result.Error("Invalid credentials provided")
        } catch (e: FirebaseAuthInvalidUserException) {
            Result.Error("User not found or session expired")
        } catch (e: FirebaseAuthException) {
            Result.Error("Firebase authentication error: ${e.message}")
        } catch (e: Exception) {
            Result.Error("Failed to update user: ${e.message}")
        }
    }



    suspend fun getUser(): Result<UserModelFirebase> {
        return try {
            val currentUser = auth.currentUser
            if (currentUser != null) {
                val user = UserModelFirebase(
                    name = currentUser.displayName ?: "",
                    email = currentUser.email ?: "",
                    password = "" // Password tidak dikembalikan untuk alasan keamanan
                )
                Result.Success(user)
            } else {
                Result.Error("No user is logged in")
            }
        } catch (e: Exception) {
            Result.Error(e.message ?: "Failed to fetch user")
        }
    }

    fun logoutUser(context : Context, onSuccess : () -> Unit, onFailure : (Exception) -> Unit) {
        auth.signOut()
        CoroutineScope(Dispatchers.IO).launch {
            try {
                // Akses UserPreferences menggunakan context
                val userPreferences = UserPreferences.getInstance(context.dataStore)
                userPreferences.logout()

                CoroutineScope(Dispatchers.Main).launch {
                    onSuccess()
                }
            } catch (e : Exception) {
                CoroutineScope(Dispatchers.Main).launch {
                    onFailure(e)
                }
            }
        }
    }
}