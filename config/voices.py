VOICES: dict[str, dict[str, list[str] | str]] = {
    "en": {
        "female": ["af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky"],
        "male": ["am_adam", "am_michael", "am_echo", "am_eric"],
        "default": "af_heart",
    },
    "es": {
        "female": ["ef_dora"],
        "male": ["em_alex", "em_santa"],
        "default": "ef_dora",
    },
}

# Kokoro language codes
LANGUAGE_CODES: dict[str, str] = {
    "en": "en-us",
    "es": "es",
}
