package com.capstone_bangkit.nutrise.userpref

import android.content.Context
import androidx.datastore.core.DataStore
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import com.capstone_bangkit.nutrise.ui.auth.firebase.model.UserModelFirebase
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map

val Context.dataStore : DataStore<Preferences> by preferencesDataStore(name = "session")

class UserPreferences private constructor(private val dataStore : DataStore<Preferences>) {

    fun getThemeSetting() : Flow<Boolean> {
        return dataStore.data.map { preferences ->
            preferences[THEME_KEY] ?: false
        }
    }

    suspend fun saveThemeSetting(isDarkModeActive: Boolean) {
        dataStore.edit { preferences ->
            preferences[THEME_KEY] = isDarkModeActive
        }
    }

    suspend fun saveSession(user : UserModelFirebase) {
        dataStore.edit { preferences ->
            preferences[NAME_KEY] = user.name.toString()
            preferences[EMAIL_KEY] = user.email.toString()
        }
    }

    fun getSession() : Flow<UserModelFirebase> {
        return dataStore.data.map { preferences ->
            UserModelFirebase(
                preferences[NAME_KEY] ?: "",
                preferences[EMAIL_KEY] ?: ""
            )
        }
    }

    suspend fun logout() {
        dataStore.edit { preferences ->
            preferences.clear()
        }
    }

    companion object {
        private var INSTANCE : UserPreferences? = null

        private val NAME_KEY = stringPreferencesKey("name")
        private val EMAIL_KEY = stringPreferencesKey("email")
        private val THEME_KEY = booleanPreferencesKey("theme_setting")

        fun getInstance(dataStore : DataStore<Preferences>) : UserPreferences {
            return INSTANCE ?: synchronized(this) {
                val instance = UserPreferences(dataStore)
                INSTANCE = instance
                instance
            }
        }
    }
}