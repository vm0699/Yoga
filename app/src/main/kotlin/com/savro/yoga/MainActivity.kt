package com.savro.yoga

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import android.speech.tts.TextToSpeech
import android.view.View
import android.view.ViewGroup
import android.view.WindowManager
import android.webkit.JavascriptInterface
import android.webkit.WebChromeClient
import android.webkit.WebResourceRequest
import android.webkit.WebResourceResponse
import android.webkit.WebSettings
import android.webkit.WebView
import android.webkit.WebViewClient
import androidx.activity.ComponentActivity
import androidx.activity.OnBackPressedCallback
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.view.ViewCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.webkit.WebViewAssetLoader
import java.util.Locale

class MainActivity : ComponentActivity(), TextToSpeech.OnInitListener {
    private lateinit var webView: WebView
    private val notifPermission =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { /* nothing to do: the alarm is already set */ }
    private var insetTop = 0
    private var insetBottom = 0
    private var insetLeft = 0
    private var insetRight = 0

    // Spoken pose/breathing cues use Android's own TextToSpeech engine, not the WebView's speechSynthesis -
    // that Web API is unreliable inside a plain WebView (often just missing) even though it works in Chrome.
    private var tts: TextToSpeech? = null
    private var ttsReady = false

    // The page is drawn edge-to-edge, so tell it how big the system bars / cutout are (in CSS px).
    private fun pushInsets() {
        if (!::webView.isInitialized) return
        val d = resources.displayMetrics.density
        val js = "var s=document.documentElement.style;" +
            "s.setProperty('--sat','${insetTop / d}px');" +
            "s.setProperty('--sab','${insetBottom / d}px');" +
            "s.setProperty('--sal','${insetLeft / d}px');" +
            "s.setProperty('--sar','${insetRight / d}px');"
        webView.evaluateJavascript(js, null)
    }

    @SuppressLint("SetJavaScriptEnabled")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        WindowCompat.setDecorFitsSystemWindows(window, false)
        tts = TextToSpeech(this, this)

        // Serve app/src/main/assets/ over a virtual https:// origin instead of a raw file:// URL.
        // Plain file:// pages in WebView block ES module scripts and fetch() (needed by the 3D pose
        // avatar's Three.js/GLTF loading and its pose-data JSON) under Chromium's same-origin rules -
        // WebViewAssetLoader is Android's documented fix, making local assets behave like a normal site.
        val assetLoader = WebViewAssetLoader.Builder()
            .addPathHandler("/assets/", WebViewAssetLoader.AssetsPathHandler(this))
            .build()

        webView = WebView(this).apply {
            settings.apply {
                javaScriptEnabled = true
                domStorageEnabled = true
                allowFileAccess = true
                allowContentAccess = true
                useWideViewPort = true
                loadWithOverviewMode = true
                cacheMode = WebSettings.LOAD_NO_CACHE
                mediaPlaybackRequiresUserGesture = false
            }
            webChromeClient = WebChromeClient()
            webViewClient = object : WebViewClient() {
                override fun shouldInterceptRequest(view: WebView, request: WebResourceRequest): WebResourceResponse? =
                    assetLoader.shouldInterceptRequest(request.url)
                override fun onPageFinished(view: WebView?, url: String?) {
                    pushInsets()
                }
            }
            setLayerType(View.LAYER_TYPE_HARDWARE, null)
            addJavascriptInterface(WebAppInterface(this@MainActivity), "AndroidBridge")
            loadUrl("https://appassets.androidplatform.net/assets/yoga.html")
        }

        setContentView(webView)

        ViewCompat.setOnApplyWindowInsetsListener(webView) { v, insets ->
            val bars = insets.getInsets(
                WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.displayCutout()
            )
            val imeVisible = insets.isVisible(WindowInsetsCompat.Type.ime())
            val imeBottom = if (imeVisible) insets.getInsets(WindowInsetsCompat.Type.ime()).bottom else 0
            insetTop = bars.top
            insetBottom = if (imeVisible) 0 else bars.bottom
            insetLeft = bars.left
            insetRight = bars.right
            // Shrink the page above the keyboard so focused inputs stay visible.
            (v.layoutParams as? ViewGroup.MarginLayoutParams)?.let { lp ->
                if (lp.bottomMargin != imeBottom) {
                    lp.bottomMargin = imeBottom
                    v.layoutParams = lp
                }
            }
            pushInsets()
            insets
        }
        WindowCompat.getInsetsController(window, webView).apply {
            isAppearanceLightStatusBars = false
            isAppearanceLightNavigationBars = true
        }

        onBackPressedDispatcher.addCallback(this, object : OnBackPressedCallback(true) {
            override fun handleOnBackPressed() {
                // Let the page handle back first (closes sheets, leaves the player, etc.).
                webView.evaluateJavascript("(window.__yogaBack&&window.__yogaBack())?'1':'0'") { handled ->
                    if (handled?.trim('"') != "1") {
                        isEnabled = false
                        onBackPressedDispatcher.onBackPressed()
                    }
                }
            }
        })
    }

    // TextToSpeech.OnInitListener
    override fun onInit(status: Int) {
        if (status != TextToSpeech.SUCCESS) { ttsReady = false; return }
        val result = tts?.setLanguage(Locale.US) ?: TextToSpeech.LANG_NOT_SUPPORTED
        ttsReady = if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
            // Fall back to whatever language the device has rather than failing outright.
            tts?.setLanguage(Locale.getDefault())
            true
        } else true
    }

    override fun onDestroy() {
        tts?.stop()
        tts?.shutdown()
        super.onDestroy()
    }

    inner class WebAppInterface(private val context: Context) {
        @JavascriptInterface
        fun vibrate(milliseconds: Long) {
            try {
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                    val manager = context.getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as? VibratorManager
                    manager?.defaultVibrator?.vibrate(
                        VibrationEffect.createOneShot(milliseconds, VibrationEffect.DEFAULT_AMPLITUDE)
                    )
                } else {
                    @Suppress("DEPRECATION")
                    val v = context.getSystemService(Context.VIBRATOR_SERVICE) as? Vibrator
                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                        v?.vibrate(VibrationEffect.createOneShot(milliseconds, VibrationEffect.DEFAULT_AMPLITUDE))
                    } else {
                        @Suppress("DEPRECATION")
                        v?.vibrate(milliseconds)
                    }
                }
            } catch (_: Exception) {}
        }

        // Daily practice reminder. On Android 13+ the notification permission is requested when it is switched on.
        @JavascriptInterface
        fun setReminder(enabled: Boolean, hour: Int, minute: Int) {
            Reminder.set(context, enabled, hour, minute)
            if (enabled && Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                runOnUiThread {
                    if (checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED) {
                        notifPermission.launch(Manifest.permission.POST_NOTIFICATIONS)
                    }
                }
            }
        }

        // Keep the screen awake while a session is playing.
        @JavascriptInterface
        fun keepAwake(on: Boolean) {
            runOnUiThread {
                if (on) window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
                else window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
            }
        }

        // Spoken pose/breathing cues, via Android's own TTS engine (see the comment on `tts` above).
        @JavascriptInterface
        fun speak(text: String) {
            if (!ttsReady) return
            runOnUiThread { tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, "yf") }
        }

        @JavascriptInterface
        fun stopSpeak() {
            runOnUiThread { tts?.stop() }
        }

        @JavascriptInterface
        fun ttsAvailable(): Boolean = ttsReady
    }
}
