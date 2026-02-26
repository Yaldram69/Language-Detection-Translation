import pyaudio
import wave
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, AutoModelForSequenceClassification, \
    AutoTokenizer, M2M100ForConditionalGeneration, M2M100Tokenizer
import threading

# Define language map
language_map = {
    'ar': 'Arabic',
    'de': 'German',
    'en': 'English',
    'es': 'Spanish',
    'fr': 'French',
    'ru': 'Russian',
    'ur': 'Urdu',
    'zh': 'Chinese'
}


# Define model loading functions
def load_language_detection_model(model_path=None):
    model = AutoModelForSequenceClassification.from_pretrained(
        "papluca/xlm-roberta-base-language-detection", cache_dir=model_path
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "papluca/xlm-roberta-base-language-detection", cache_dir=model_path
    )
    return pipeline("text-classification", model=model, tokenizer=tokenizer)


def load_translation_model(model_path=None):
    model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M", cache_dir=model_path)
    tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M", cache_dir=model_path)
    return model, tokenizer


# Audio Transcriber class
class AudioTranscriber:
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model_id = "openai/whisper-large-v3"
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        self.model.to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )

    def transcribe_audio(self, audio_path):
        results = self.pipe(audio_path)
        return results["text"]


# Audio Recorder class (Modified for indefinite recording until Enter is pressed)
class AudioRecorder:
    def __init__(self, sample_rate=16000, channels=1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.recording = False
        self.frames = []

    def start_recording(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=self.format,
                        channels=self.channels,
                        rate=self.sample_rate,
                        input=True,
                        frames_per_buffer=self.chunk)
        print("Recording... Press Enter to stop.")
        self.recording = True

        while self.recording:
            data = stream.read(self.chunk)
            self.frames.append(data)

        stream.stop_stream()
        stream.close()
        p.terminate()

    def stop_recording(self):
        self.recording = False

    def save_audio(self, filename):
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(pyaudio.PyAudio().get_sample_size(self.format))
        wf.setframerate(self.sample_rate)
        wf.writeframes(b''.join(self.frames))
        wf.close()
        print(f"Audio saved as {filename}")


# Language Detector class
class LanguageDetector:
    def __init__(self, model_path=None, language_map=None):
        self.model_path = model_path
        self.language_map = language_map
        self.language_detector = load_language_detection_model(self.model_path)

    def detect_language(self, text):
        result = self.language_detector(text)
        language_code = result[0]['label']
        return self.language_map.get(language_code, language_code)


# Translator class
class Translator:
    def __init__(self, model_path=None):
        self.translation_model, self.tokenizer = load_translation_model(model_path)

    def translate_to_english(self, text, source_language):
        self.tokenizer.src_lang = source_language
        encoded_text = self.tokenizer(text, return_tensors="pt")
        generated_tokens = self.translation_model.generate(**encoded_text,
                                                           forced_bos_token_id=self.tokenizer.get_lang_id("en"))
        return self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]


# Main function to run the process
def main():
    # Initialize components
    recorder = AudioRecorder()
    transcriber = AudioTranscriber()
    detector = LanguageDetector(model_path="D:/PythonProjects/Text Detection/models/xlm-roberta",
                                language_map=language_map)
    translator = Translator(model_path="D:/PythonProjects/Text Detection/models/m2m100")

    # Start recording in a separate thread
    def record_thread():
        recorder.start_recording()

    record_thread = threading.Thread(target=record_thread)
    record_thread.start()

    # Wait for user to press Enter to stop recording
    input("Press Enter to stop recording...")

    # Stop the recording
    recorder.stop_recording()

    # Save audio
    audio_file = "recording.wav"
    recorder.save_audio(audio_file)

    # Transcribe audio
    transcription = transcriber.transcribe_audio(audio_file)
    print("Transcribed text:")
    print(transcription)

    # Detect language
    detected_language = detector.detect_language(transcription)
    print(f"Detected Language: {detected_language}")

    # Translate to English
    if detected_language.lower() != 'english':
        source_language_code = list(language_map.keys())[list(language_map.values()).index(detected_language)]
        translation = translator.translate_to_english(transcription, source_language_code)
        print(f"Translation to English: {translation}")
    else:
        print("No translation needed, the text is already in English.")


if __name__ == "__main__":
    main()
