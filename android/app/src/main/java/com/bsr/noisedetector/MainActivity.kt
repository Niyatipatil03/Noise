package com.bsr.noisedetector

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import android.view.View
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.bsr.noisedetector.databinding.ActivityMainBinding
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*
import java.text.SimpleDateFormat
import java.util.*

/**
 * Main screen — controls recording and shows live detection status.
 *
 * Workflow:
 *  1. User taps "START ANALYSIS"
 *  2. App connects to Bluetooth mic (or falls back to internal mic)
 *  3. Every 3-second window is classified in real-time
 *  4. Results update live on screen
 *  5. Tap "STOP" → results summary → optional "SAVE REPORT"
 */
class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var recorder: AudioRecorder
    private lateinit var classifier: NoiseClassifier

    private val detectionHistory = mutableListOf<NoiseClassifier.Result>()
    private var analysisJob: Job? = null
    private var windowCount = 0

    // ── Permissions ───────────────────────────────────────────────────────────

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { grants ->
        if (grants.all { it.value }) {
            startAnalysis()
        } else {
            Toast.makeText(this, "Microphone permission is required", Toast.LENGTH_LONG).show()
        }
    }

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        classifier = NoiseClassifier(this)
        recorder   = AudioRecorder(this)

        setupUI()
        observeRecorderState()
    }

    override fun onDestroy() {
        super.onDestroy()
        recorder.release()
        classifier.close()
    }

    // ── UI setup ──────────────────────────────────────────────────────────────

    private fun setupUI() {
        binding.btnStartStop.setOnClickListener {
            if (recorder.state.value == AudioRecorder.State.RECORDING) {
                stopAnalysis()
            } else {
                requestPermissionsAndStart()
            }
        }

        binding.btnViewResults.setOnClickListener {
            if (detectionHistory.isNotEmpty()) showResultsSummary()
        }

        binding.btnViewResults.visibility = View.GONE
    }

    private fun observeRecorderState() {
        lifecycleScope.launch {
            recorder.state.collect { state ->
                when (state) {
                    AudioRecorder.State.IDLE -> {
                        binding.btnStartStop.text = "START ANALYSIS"
                        binding.btnStartStop.isEnabled = true
                        binding.progressBar.visibility = View.GONE
                    }
                    AudioRecorder.State.CONNECTING_BT -> {
                        binding.btnStartStop.text = "Connecting…"
                        binding.btnStartStop.isEnabled = false
                        binding.progressBar.visibility = View.VISIBLE
                        binding.tvStatus.text = "Connecting to Bluetooth mic…"
                    }
                    AudioRecorder.State.RECORDING -> {
                        binding.btnStartStop.text = "STOP"
                        binding.btnStartStop.isEnabled = true
                        binding.progressBar.visibility = View.GONE
                        val src = if (recorder.bluetoothConnected.value) "Bluetooth mic" else "Internal mic"
                        binding.tvMicSource.text = "Source: $src"
                    }
                    AudioRecorder.State.ERROR -> {
                        binding.btnStartStop.text = "START ANALYSIS"
                        binding.btnStartStop.isEnabled = true
                        binding.progressBar.visibility = View.GONE
                        Toast.makeText(this@MainActivity, "Recording error", Toast.LENGTH_SHORT).show()
                    }
                }
            }
        }
    }

    // ── Analysis flow ─────────────────────────────────────────────────────────

    private fun requestPermissionsAndStart() {
        val needed = mutableListOf<String>()
        if (!hasPermission(Manifest.permission.RECORD_AUDIO))
            needed.add(Manifest.permission.RECORD_AUDIO)
        if (!hasPermission(Manifest.permission.BLUETOOTH_CONNECT) &&
            android.os.Build.VERSION.SDK_INT >= 31)
            needed.add(Manifest.permission.BLUETOOTH_CONNECT)

        if (needed.isEmpty()) startAnalysis()
        else permissionLauncher.launch(needed.toTypedArray())
    }

    private fun startAnalysis() {
        detectionHistory.clear()
        windowCount = 0
        binding.btnViewResults.visibility = View.GONE
        updateResultCard(null)

        recorder.startRecording()

        analysisJob = lifecycleScope.launch {
            recorder.audioClips.collect { waveform ->
                withContext(Dispatchers.Default) {
                    val result = classifier.classify(waveform)
                    withContext(Dispatchers.Main) {
                        detectionHistory.add(result)
                        windowCount++
                        updateResultCard(result)
                        updateStats()
                    }
                }
            }
        }
    }

    private fun stopAnalysis() {
        analysisJob?.cancel()
        recorder.stopRecording()
        binding.btnViewResults.visibility =
            if (detectionHistory.isNotEmpty()) View.VISIBLE else View.GONE
    }

    // ── Live result display ───────────────────────────────────────────────────

    private fun updateResultCard(result: NoiseClassifier.Result?) {
        if (result == null) {
            binding.tvStatus.text = "Waiting for audio…"
            binding.cardResult.setCardBackgroundColor(getColor(R.color.status_idle))
            binding.tvNoiseType.text = ""
            binding.tvConfidence.text = ""
            return
        }

        val notOkPct = (result.notOkProbability * 100).toInt()
        val okPct    = 100 - notOkPct
        val timestamp = SimpleDateFormat("HH:mm:ss", Locale.getDefault()).format(Date())

        if (result.isNotOk) {
            binding.tvStatus.text = "⚠  NOT OK  [$timestamp]"
            binding.cardResult.setCardBackgroundColor(getColor(R.color.status_not_ok))
            binding.tvNoiseType.text = "Noise: ${result.noiseTypeName}"
            binding.tvConfidence.text = "Confidence: $notOkPct% NOT_OK"
        } else {
            binding.tvStatus.text = "✓  OK  [$timestamp]"
            binding.cardResult.setCardBackgroundColor(getColor(R.color.status_ok))
            binding.tvNoiseType.text = ""
            binding.tvConfidence.text = "Confidence: $okPct% OK"
        }
    }

    private fun updateStats() {
        val total   = detectionHistory.size
        val notOkCt = detectionHistory.count { it.isNotOk }
        val okCt    = total - notOkCt
        binding.tvStats.text = "Windows: $total  |  OK: $okCt  |  NOT OK: $notOkCt"
    }

    private fun showResultsSummary() {
        val intent = Intent(this, ResultsActivity::class.java).apply {
            putExtra(ResultsActivity.EXTRA_TOTAL,   detectionHistory.size)
            putExtra(ResultsActivity.EXTRA_NOT_OK,  detectionHistory.count { it.isNotOk })
            // Pass top noise types as a string summary
            val typeFreq = detectionHistory
                .filter { it.isNotOk }
                .groupBy { it.noiseTypeName }
                .mapValues { it.value.size }
                .entries.sortedByDescending { it.value }
            val typeSummary = typeFreq.joinToString("|") { "${it.key}:${it.value}" }
            putExtra(ResultsActivity.EXTRA_NOISE_TYPES, typeSummary)
        }
        startActivity(intent)
    }

    private fun hasPermission(perm: String) =
        ContextCompat.checkSelfPermission(this, perm) == PackageManager.PERMISSION_GRANTED
}
