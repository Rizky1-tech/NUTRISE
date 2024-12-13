package com.capstone_bangkit.nutrise.ui.setting.history

import android.os.Bundle
import android.view.View
import androidx.activity.viewModels
import androidx.appcompat.app.AppCompatActivity
import com.capstone_bangkit.nutrise.R
import com.capstone_bangkit.nutrise.databinding.ActivityHistoryBinding

class HistoryActivity : AppCompatActivity() {

    private lateinit var binding : ActivityHistoryBinding
    private val historyViewModel : HistoryViewModel by viewModels()
    private lateinit var historyAdapter : HistoryAdapter

    override fun onCreate(savedInstanceState : Bundle?) {
        super.onCreate(savedInstanceState)

        // Inflate layout dan set sebagai konten tampilan
        binding = ActivityHistoryBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // Set up toolbar
        setSupportActionBar(binding.toolbar)
        supportActionBar?.setDisplayHomeAsUpEnabled(true) // Menambahkan tombol kembali

        // Set up RecyclerView
        historyAdapter = HistoryAdapter(emptyList())
        binding.recyclerViewHistory.apply {
            layoutManager = androidx.recyclerview.widget.LinearLayoutManager(this@HistoryActivity)
            adapter = historyAdapter
        }

        // Observe LiveData dari ViewModel
        historyViewModel.allHistory.observe(this) { historyList ->
            if (historyList.isNotEmpty()) {
                historyAdapter.updateData(historyList)
            } else {
                // Tampilkan pesan jika tidak ada data
                binding.tvEmptyMessage.visibility = View.VISIBLE
                binding.recyclerViewHistory.visibility = View.GONE
            }
        }
    }

    // Menambahkan opsi menu pada toolbar
    override fun onCreateOptionsMenu(menu : android.view.Menu?) : Boolean {
        menuInflater.inflate(R.menu.history_menu, menu)
        return true
    }

    // Menambahkan aksi untuk menghapus history saat ikon di toolbar ditekan
    override fun onOptionsItemSelected(item : android.view.MenuItem) : Boolean {
        return when (item.itemId) {
            R.id.action_delete_history -> {
                // Menghapus semua history
                historyViewModel.deleteAllHistory()
                true
            }

            android.R.id.home -> {
                // Navigasi kembali ke activity sebelumnya
                @Suppress("DEPRECATION")
                onBackPressed()
                true
            }

            else -> super.onOptionsItemSelected(item)
        }
    }
}


