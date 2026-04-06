package com.bsr.noisedetector

import android.content.Intent
import android.os.Bundle
import android.os.Environment
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.bsr.noisedetector.databinding.ActivityResultsBinding
import java.io.File
import java.text.SimpleDateFormat
import java.util.*

/**
 * Shows a summary of the completed BSR analysis session and allows
 * saving a plain-text report to device storage.
 */
class ResultsActivity : AppCompatActivity() {

    companion object {
        const val EXTRA_TOTAL       = "extra_total"
        const val EXTRA_NOT_OK      = "extra_not_ok"
        const val EXTRA_NOISE_TYPES = "extra_noise_types"
    }

    private lateinit var binding: ActivityResultsBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityResultsBinding.inflate(layoutInflater)
        setContentView(binding.root)

        val total      = intent.getIntExtra(EXTRA_TOTAL, 0)
        val notOkCount = intent.getIntExtra(EXTRA_NOT_OK, 0)
        val okCount    = total - notOkCount
        val typesRaw   = intent.getStringExtra(EXTRA_NOISE_TYPES) ?: ""

        // Parse noise types summary
        val typeEntries = if (typesRaw.isBlank()) emptyList()
        else typesRaw.split("|").map { entry ->
            val parts = entry.split(":")
            parts.getOrElse(0) { "Unknown" } to (parts.getOrElse(1) { "0" }.toIntOrNull() ?: 0)
        }

        displayResults(total, okCount, notOkCount, typeEntries)

        binding.btnSaveReport.setOnClickListener {
            saveReport(total, okCount, notOkCount, typeEntries)
        }

        binding.btnBackHome.setOnClickListener { finish() }
    }

    private fun displayResults(
        total: Int,
        okCount: Int,
        notOkCount: Int,
        typeEntries: List<Pair<String, Int>>,
    ) {
        val overallStatus = if (notOkCount > okCount) "NOT OK ⚠" else "OK ✓"
        binding.tvOverallStatus.text = overallStatus
        binding.tvOverallStatus.setTextColor(
            getColor(if (notOkCount > okCount) R.color.status_not_ok else R.color.status_ok)
        )

        binding.tvTotalWindows.text  = "Total windows analysed : $total"
        binding.tvOkWindows.text     = "OK windows             : $okCount"
        binding.tvNotOkWindows.text  = "NOT OK windows         : $notOkCount"

        val notOkPct = if (total > 0) (notOkCount * 100f / total).toInt() else 0
        binding.tvNotOkPercent.text  = "NOT OK rate            : $notOkPct%"
        binding.progressNotOk.progress = notOkPct

        // Noise types breakdown
        if (typeEntries.isEmpty()) {
            binding.tvNoiseBreakdown.text = "No noise events detected."
        } else {
            val sb = StringBuilder("Noise Type Breakdown:\n")
            typeEntries.forEach { (name, count) ->
                val pct = if (notOkCount > 0) (count * 100 / notOkCount) else 0
                sb.append("  • $name : $count windows ($pct%)\n")
            }
            binding.tvNoiseBreakdown.text = sb.toString()
        }
    }

    private fun saveReport(
        total: Int, okCount: Int, notOkCount: Int,
        typeEntries: List<Pair<String, Int>>,
    ) {
        val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
        val report = buildReportText(total, okCount, notOkCount, typeEntries, timestamp)

        try {
            val dir = getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS)
                ?: filesDir
            val file = File(dir, "BSR_Report_$timestamp.txt")
            file.writeText(report)
            Toast.makeText(this, "Report saved: ${file.name}", Toast.LENGTH_LONG).show()
        } catch (e: Exception) {
            Toast.makeText(this, "Save failed: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }

    private fun buildReportText(
        total: Int, okCount: Int, notOkCount: Int,
        typeEntries: List<Pair<String, Int>>,
        timestamp: String,
    ): String = buildString {
        appendLine("═══════════════════════════════════════")
        appendLine("  BSR Noise Detection Report")
        appendLine("  Timestamp : $timestamp")
        appendLine("═══════════════════════════════════════")
        appendLine()
        appendLine("Overall Status  : ${if (notOkCount > okCount) "NOT OK" else "OK"}")
        appendLine("Total windows   : $total")
        appendLine("OK windows      : $okCount")
        appendLine("NOT OK windows  : $notOkCount")
        val pct = if (total > 0) (notOkCount * 100f / total).toInt() else 0
        appendLine("NOT OK rate     : $pct%")
        appendLine()
        if (typeEntries.isNotEmpty()) {
            appendLine("Noise Type Breakdown:")
            typeEntries.forEach { (name, count) ->
                appendLine("  • $name : $count windows")
            }
        } else {
            appendLine("No noise detected.")
        }
        appendLine()
        appendLine("Generated by BSR Noise Detector v1.0")
        appendLine("═══════════════════════════════════════")
    }
}
