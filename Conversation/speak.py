from kokoro import KPipeline
import sounddevice as sd

_PIPELINES = {}

def speak(text, lang="b", voice="bm_lewis", samplerate=24000, speed=1.0):
    if lang not in _PIPELINES:
        _PIPELINES[lang] = KPipeline(lang_code=lang)

    pipeline = _PIPELINE[lang]

    for _, _, audio in pipeline(text, voice=voice):
        sd.play(audio, samplerate=speed * samplerate)
        sd.wait()
