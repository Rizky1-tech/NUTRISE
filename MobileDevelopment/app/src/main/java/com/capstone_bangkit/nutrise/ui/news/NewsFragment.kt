package com.capstone_bangkit.nutrise.ui.news

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.Fragment
import androidx.lifecycle.ViewModelProvider
import androidx.recyclerview.widget.LinearLayoutManager
import com.capstone_bangkit.nutrise.R
import com.capstone_bangkit.nutrise.databinding.FragmentNewsBinding

class NewsFragment : Fragment() {

    private var _binding: FragmentNewsBinding? = null
    private val binding get() = _binding!!

    private lateinit var newsAdapter: NewsAdapter

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        val newsViewModel =
            ViewModelProvider(this).get(NewsViewModel::class.java)

        _binding = FragmentNewsBinding.inflate(inflater, container, false)
        val root: View = binding.root

        // Set up RecyclerView and Adapter
        newsAdapter = NewsAdapter(getNewsData())
        binding.recycleApiUpcoming.layoutManager = LinearLayoutManager(context)
        binding.recycleApiUpcoming.adapter = newsAdapter

        return root
    }

    private fun getNewsData(): List<News> {
        return listOf(
            News("Sederet Manfaat Kesehatan Biji Ketumbar, Ampuh Turunkan Kolesterol", "Biji ketumbar berasal dari tumbuhan ketumbar (Coriandrum sativum) yang masih punya hubungan dengan parsley, wortel dan seledri. Di Indonesia, biji ketumbar umum digunakan sebagai bumbu masakan.", R.drawable.bijiketumbar),
            News("5 Manfaat Daun Jambu Biji, Tak Cuma Atasi Diare", "Manfaat daun jambu biji yang paling populer bisa jadi untuk meredakan diare. Padahal daun ini mampu memberikan aneka manfaat kesehatan.", R.drawable.daunjambubiji),
            News("Cukup 30 Menit, Rasakan 5 Manfaat Berjalan Kaki Ruin Tiap Hari", "Jalan kaki jadi olahraga murah dan mudah dilakukan. Jika dilakukan rutin setiap hari, manfaat berjalan kaki selama 30 menit ada banyak.", R.drawable.jalankaki),
            News("Mengulik soal Kanker Prostat, Bahaya yang Kerap Tak Disadari", "Kanker prostat adalah penyakit yang kerap dianggap momok bagi para lelaki. Sebenarnya, kanker prostat dapat ditangani dan hasilnya akan lebih baik jika terdeteksi dini.", R.drawable.kanker),
            News("Mau Awet Muda? Rutin Lakukan OlahRaga Sederhana ini", "Tampil awet muda tak hanya bermodal rajin pakai produk skincare. Tubuh pun perlu dirawat dengan aktivitas fisik rutin.", R.drawable.olahraga),
            News("Viral di TikTok, Apa itu Diet 90–30-50?", "Jika Anda kerap gagal dengan sejumlah metode diet, mungkin Anda perlu mencoba diet 90-30-50 yang sedang viral di TikTok.", R.drawable.timbangan)
        )
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}