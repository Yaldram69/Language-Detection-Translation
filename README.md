# Language Detection and Translation (Speech)

A command-line tool that records speech from the microphone, transcribes it with Whisper, detects the language of the transcript, and translates it to English when it is not already English. It uses `openai/whisper-large-v3` for speech recognition, `papluca/xlm-roberta-base-language-detection` for language detection, and `facebook/m2m100_418M` for translation.

## Features

- Microphone recording with PyAudio, 16 kHz mono, until the user presses Enter.
- Speech-to-text with Whisper large-v3 (GPU in FP16 when CUDA is available, otherwise CPU).
- Language detection, mapped to 8 supported languages: Arabic, German, English, Spanish, French, Russian, Urdu, Chinese.
- Translation to English with M2M100 when the detected language is not English.

## Project structure

- `audiotest1.py` - main script. Records audio, saves `recording.wav`, transcribes it, detects the language, and translates.
- `ReadMe.txt` - original run instructions.
- `recording.wav` - sample recording (the file the script writes on each run).
- `real_time_recording.wav` - another sample recording.
- `DEMO.mp4` - demo video.
- `models/` - not included (see Notes). Holds the cached `xlm-roberta` and `m2m100` models on the original machine.

## Requirements

Imports seen in the code: `pyaudio`, `torch`, `transformers` (Whisper, XLM-RoBERTa and M2M100 classes), and the standard library modules `wave` and `threading`. `wave` is part of Python, so it does not need `pip install`. M2M100 tokenization needs `sentencepiece`.

```bash
pip install pyaudio torch transformers sentencepiece
```

PyAudio on Windows may need a prebuilt wheel if `pip install pyaudio` fails.

## Installation

1. Install the Python packages above.
2. Install ffmpeg. The Whisper pipeline reads the WAV file through ffmpeg, so it must be available on the system `PATH`. The original project shipped an ffmpeg build in `ffmpeg-master-latest-win64-gpl/bin`; that folder is not included. Download ffmpeg from https://ffmpeg.org and add its `bin` folder to `PATH`.
3. Clone the repository and change into it:

```bash
git clone <this-repo-url>
cd Language-Detection-Translation
```

4. Edit the `main()` function in `audiotest1.py`. The model cache paths are hard-coded to `D:/PythonProjects/Text Detection/models/...`; change them to a folder you control.

## Usage

```bash
python audiotest1.py
```

The script prints `Recording... Press Enter to stop.`. Speak in any supported language, press Enter, and the script prints the transcript, the detected language and the English translation (when needed). It writes `recording.wav` to the current folder on each run, replacing the previous file.

## Notes

- **Excluded: `models/` folder.** It holds several GB of Hugging Face weights (`facebook/m2m100_418M`, `papluca/xlm-roberta-base-language-detection`). These were not copied to the repository because of their size. The code downloads them with `from_pretrained(..., cache_dir=...)` on first run, and Whisper large-v3 is also downloaded from Hugging Face on first run. Internet access is needed for the first run.
- **Excluded: `ffmpeg-master-latest-win64-gpl/`.** The bundled ffmpeg build was not copied (it is a large third-party binary). Install ffmpeg separately as described above.
- Language detection only maps 8 codes. If the detector returns a code outside this list (for example `it`), the lookup used for translation raises a `ValueError`. Extend `language_map` to avoid this.
- The recording length is set by the Enter key, not a fixed duration; `ReadMe.txt` mentions a 10-second recording, which the current code does not do.
- Audio is recorded at 16 kHz, mono, 16-bit.
