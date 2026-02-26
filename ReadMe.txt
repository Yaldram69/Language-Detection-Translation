# Language Detection & Translation (Speech → English)

A speech pipeline that:
1) records audio from microphone  
2) transcribes speech using **Whisper**  
3) detects the language from the transcription  
4) translates non-English text into **English** using **M2M100**.

---

## Features
- Mic recording (press Enter to stop)
- Speech-to-text transcription (Whisper)
- Language detection (XLM-R language detection)
- Translation to English (M2M100)

---

## Tech Stack
- Python
- PyAudio
- HuggingFace Transformers
- Whisper (openai/whisper-large-v3)
- XLM-R language detection (papluca/xlm-roberta-base-language-detection)
- M2M100 translation (facebook/m2m100_418M)
- Torch

---
