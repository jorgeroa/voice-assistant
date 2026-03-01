// Voice Conversation - WebSocket Client
// Handles mic capture, audio playback, and server communication.

const SAMPLE_RATE = 16000;
const CHUNK_DURATION_MS = 32;
const CHUNK_SIZE = 512; // power of 2 for createScriptProcessor + Silero VAD compatible

class VoiceApp {
    constructor() {
        // DOM
        this.micBtn = document.getElementById("mic-btn");
        this.statusText = document.getElementById("status-text");
        this.messagesEl = document.getElementById("messages");
        this.languageSelect = document.getElementById("language");
        this.inputModeSelect = document.getElementById("input-mode");
        this.sttModeSelect = document.getElementById("stt-mode");
        this.vadMeter = document.getElementById("vad-meter");
        this.vadFill = document.getElementById("vad-fill");

        // State
        this.ws = null;
        this.mediaStream = null;
        this.audioContext = null;
        this.scriptProcessor = null;
        this.isRecording = false;
        this.isProcessing = false;
        this.inputMode = "ptt";
        this.currentAssistantEl = null;
        this.audioQueue = [];
        this.isPlayingAudio = false;

        // Playback context (separate from capture to avoid conflicts)
        this.playbackCtx = null;

        this.init();
    }

    async init() {
        this.bindEvents();
        this.connectWebSocket();
    }

    bindEvents() {
        // Mic button: PTT uses mousedown/up, VAD uses click toggle
        this.micBtn.addEventListener("mousedown", (e) => {
            if (this.inputMode === "ptt" && !this.isProcessing) {
                e.preventDefault();
                this.startRecording();
            }
        });
        this.micBtn.addEventListener("mouseup", (e) => {
            if (this.inputMode === "ptt" && this.isRecording) {
                e.preventDefault();
                this.stopRecording();
            }
        });
        this.micBtn.addEventListener("mouseleave", () => {
            if (this.inputMode === "ptt" && this.isRecording) {
                this.stopRecording();
            }
        });
        this.micBtn.addEventListener("click", () => {
            if (this.inputMode === "vad") {
                if (this.isRecording) {
                    this.stopVADListening();
                } else {
                    this.startVADListening();
                }
            }
        });

        // Touch support for mobile PTT
        this.micBtn.addEventListener("touchstart", (e) => {
            if (this.inputMode === "ptt" && !this.isProcessing) {
                e.preventDefault();
                this.startRecording();
            }
        });
        this.micBtn.addEventListener("touchend", (e) => {
            if (this.inputMode === "ptt" && this.isRecording) {
                e.preventDefault();
                this.stopRecording();
            }
        });

        // Settings changes
        this.languageSelect.addEventListener("change", () => {
            this.send({ type: "set_language", language: this.languageSelect.value });
        });
        this.inputModeSelect.addEventListener("change", () => {
            this.inputMode = this.inputModeSelect.value;
            this.send({ type: "set_mode", mode: this.inputMode });
            this.vadMeter.classList.toggle("hidden", this.inputMode !== "vad");
            if (this.isRecording && this.inputMode === "ptt") {
                this.stopVADListening();
            }
        });
        this.sttModeSelect.addEventListener("change", () => {
            this.send({ type: "set_stt_mode", mode: this.sttModeSelect.value });
        });

        // Keyboard shortcut: Space for PTT
        document.addEventListener("keydown", (e) => {
            if (e.code === "Space" && this.inputMode === "ptt" &&
                !this.isRecording && !this.isProcessing &&
                e.target.tagName !== "SELECT") {
                e.preventDefault();
                this.startRecording();
            }
        });
        document.addEventListener("keyup", (e) => {
            if (e.code === "Space" && this.inputMode === "ptt" && this.isRecording) {
                e.preventDefault();
                this.stopRecording();
            }
        });
    }

    // --- WebSocket ---

    connectWebSocket() {
        const protocol = location.protocol === "https:" ? "wss:" : "ws:";
        const url = `${protocol}//${location.host}/ws/conversation`;
        this.ws = new WebSocket(url);

        this.ws.onopen = () => {
            this.setStatus("Connected. Ready to talk.");
            this.micBtn.disabled = false;
        };

        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleServerMessage(data);
        };

        this.ws.onclose = () => {
            this.setStatus("Disconnected. Reconnecting...");
            this.micBtn.disabled = true;
            setTimeout(() => this.connectWebSocket(), 2000);
        };

        this.ws.onerror = () => {
            this.setStatus("Connection error.");
        };
    }

    send(data) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(data));
        }
    }

    handleServerMessage(data) {
        switch (data.type) {
            case "status":
                this.setStatus(data.message);
                break;
            case "vad_status":
                this.updateVADMeter(data.probability);
                break;
            case "transcript":
                this.addUserMessage(data.text, data.speaker, data.processing_time);
                break;
            case "llm_chunk":
                this.appendAssistantChunk(data.text);
                break;
            case "audio_response":
                this.queueAudio(data.data, data.sample_rate);
                break;
            case "turn_complete":
                this.onTurnComplete();
                break;
            case "error":
                this.setStatus(`Error: ${data.message}`);
                this.setProcessing(false);
                break;
        }
    }

    // --- Audio Capture ---

    async ensureAudioContext() {
        if (this.audioContext) return;

        this.audioContext = new AudioContext({ sampleRate: SAMPLE_RATE });
        this.mediaStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                sampleRate: SAMPLE_RATE,
                channelCount: 1,
                echoCancellation: true,
                noiseSuppression: true,
            }
        });

        const source = this.audioContext.createMediaStreamSource(this.mediaStream);
        // ScriptProcessorNode for broad compatibility (AudioWorklet would be better for production)
        this.scriptProcessor = this.audioContext.createScriptProcessor(CHUNK_SIZE, 1, 1);

        this.scriptProcessor.onaudioprocess = (event) => {
            if (!this.isRecording) return;
            const input = event.inputBuffer.getChannelData(0);
            // Convert float32 to int16
            const int16 = new Int16Array(input.length);
            for (let i = 0; i < input.length; i++) {
                int16[i] = Math.max(-32768, Math.min(32767, Math.round(input[i] * 32768)));
            }
            const b64 = this.arrayBufferToBase64(int16.buffer);
            this.send({ type: "audio_chunk", data: b64 });
        };

        source.connect(this.scriptProcessor);
        this.scriptProcessor.connect(this.audioContext.destination);
    }

    async startRecording() {
        try {
            await this.ensureAudioContext();
            if (this.audioContext.state === "suspended") {
                await this.audioContext.resume();
            }
            this.isRecording = true;
            this.micBtn.classList.add("recording");
            this.send({ type: "start_recording" });
            this.setStatus("Listening...");
        } catch (err) {
            this.setStatus(`Mic error: ${err.message}`);
        }
    }

    stopRecording() {
        this.isRecording = false;
        this.micBtn.classList.remove("recording");
        this.send({ type: "stop_recording" });
        this.setProcessing(true);
    }

    async startVADListening() {
        try {
            await this.ensureAudioContext();
            if (this.audioContext.state === "suspended") {
                await this.audioContext.resume();
            }
            this.isRecording = true;
            this.micBtn.classList.add("recording");
            this.setStatus("Listening (VAD)...");
        } catch (err) {
            this.setStatus(`Mic error: ${err.message}`);
        }
    }

    stopVADListening() {
        this.isRecording = false;
        this.micBtn.classList.remove("recording");
        this.setStatus("Ready.");
    }

    // --- Audio Playback ---

    queueAudio(base64Data, sampleRate) {
        this.audioQueue.push({ data: base64Data, sampleRate });
        if (!this.isPlayingAudio) {
            this.playNextAudio();
        }
    }

    async playNextAudio() {
        if (this.audioQueue.length === 0) {
            this.isPlayingAudio = false;
            return;
        }
        this.isPlayingAudio = true;

        if (!this.playbackCtx) {
            this.playbackCtx = new AudioContext();
        }
        if (this.playbackCtx.state === "suspended") {
            await this.playbackCtx.resume();
        }

        const { data, sampleRate } = this.audioQueue.shift();
        const raw = this.base64ToArrayBuffer(data);
        const int16 = new Int16Array(raw);
        const float32 = new Float32Array(int16.length);
        for (let i = 0; i < int16.length; i++) {
            float32[i] = int16[i] / 32768.0;
        }

        const buffer = this.playbackCtx.createBuffer(1, float32.length, sampleRate);
        buffer.getChannelData(0).set(float32);

        const source = this.playbackCtx.createBufferSource();
        source.buffer = buffer;
        source.connect(this.playbackCtx.destination);
        source.onended = () => this.playNextAudio();
        source.start();
    }

    // --- UI Updates ---

    setStatus(text) {
        this.statusText.textContent = text;
    }

    setProcessing(isProcessing) {
        this.isProcessing = isProcessing;
        this.micBtn.classList.toggle("processing", isProcessing);
        this.micBtn.disabled = isProcessing;
    }

    updateVADMeter(probability) {
        this.vadFill.style.width = `${Math.round(probability * 100)}%`;
    }

    addUserMessage(text, speaker, processingTime) {
        const el = document.createElement("div");
        el.className = "message user";

        let metaHtml = '<span class="role">You</span>';
        if (speaker) {
            metaHtml += ` <span class="speaker">[${speaker}]</span>`;
        }
        if (processingTime) {
            metaHtml += ` <span class="time">${processingTime}s</span>`;
        }

        el.innerHTML = `<div class="meta">${metaHtml}</div><div class="content">${this.escapeHtml(text)}</div>`;
        this.messagesEl.appendChild(el);
        this.scrollToBottom();
    }

    appendAssistantChunk(text) {
        if (!this.currentAssistantEl) {
            this.currentAssistantEl = document.createElement("div");
            this.currentAssistantEl.className = "message assistant streaming";
            this.currentAssistantEl.innerHTML =
                '<div class="meta"><span class="role">Assistant</span></div><div class="content"></div>';
            this.messagesEl.appendChild(this.currentAssistantEl);
        }
        const contentEl = this.currentAssistantEl.querySelector(".content");
        contentEl.textContent += text;
        this.scrollToBottom();
    }

    onTurnComplete() {
        if (this.currentAssistantEl) {
            this.currentAssistantEl.classList.remove("streaming");
            this.currentAssistantEl = null;
        }
        this.setProcessing(false);
        this.setStatus("Ready.");
    }

    scrollToBottom() {
        const main = document.querySelector("main");
        main.scrollTop = main.scrollHeight;
    }

    // --- Utilities ---

    escapeHtml(text) {
        const div = document.createElement("div");
        div.textContent = text;
        return div.innerHTML;
    }

    arrayBufferToBase64(buffer) {
        const bytes = new Uint8Array(buffer);
        let binary = "";
        for (let i = 0; i < bytes.byteLength; i++) {
            binary += String.fromCharCode(bytes[i]);
        }
        return btoa(binary);
    }

    base64ToArrayBuffer(base64) {
        const binary = atob(base64);
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i++) {
            bytes[i] = binary.charCodeAt(i);
        }
        return bytes.buffer;
    }
}

// Boot
document.addEventListener("DOMContentLoaded", () => {
    new VoiceApp();
});
