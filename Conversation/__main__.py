from speak import speak 
import ollama
import whisper
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import tempfile
import os

SAMPLE_RATE = 16000
SILENCE_THRESHOLD = 500
SILENCE_LIMIT = 1.0

def record():
    print("SPEAK")

    audio_buffer = []
    silent_chunks = 0
    chunk_duration = 0.2
    chunk_size = int(SAMPLE_RATE * chunk_duration)

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16') as stream:
        while True:
            chunk, _ = stream.read(chunk_size)
            audio_buffer.append(chunk)

            volume = np.abs(chunk).mean()

            if volume < SILENCE_THRESHOLD:
                silent_chunks += 1
            else:
                silent_chunks = 0

            if silent_chunks > (SILENCE_LIMIT / chunk_duration):
                break 

    audio = np.concatenate(audio_buffer, axis=0)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        filename = f.name 
        wav.write(filename, SAMPLE_RATE, audio)
    
    return filename


def generate_response(prompt: str, history: list, model: str):
    history.append({"role": "user", "content": prompt})

    response = ollama.chat(model=model, messages=history)
    reply = response["message"]["content"]
    history.append({"role": "assistant", "content": reply})
    return reply, history

def main():
    history = []
    stt_model = whisper.load_model("small")
    model = os.environ["MODEL"]
    if model is None: model = "llama3.2:1b"

    while True:
        audio_file = record()

        prompt = stt_model.transcribe(audio_file)["text"].strip()
        print("You: ", prompt)

        # sentinel
        if prompt.lower() in ["quit", "exit", "stop"]: break

        response, history = generate_response(prompt, history, model=model)
        print("Assistant: ", response)
        speak(response, lang="a", voice="af_heart", speed=1)

if __name__ == "__main__":
    main()
