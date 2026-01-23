from TTS.tts.utils.text.phonemizers.espeak_wrapper import ESpeak

phonemizer = ESpeak(language="de")  # oder "de-DE"
text = "Huhu. Der Tag ist vorüber und die Nacht bricht herein."
phonemes = phonemizer.phonemize(text)  # gibt echte Phoneme zurück
print(phonemes)

# optional:

model_path="/home/projects/vokquant/coqui/results/vits/wass_all/vits_wass-de-November-18-2025_09+07AM-c0749fb6/best_model.pth"
config_path="/home/projects/vokquant/coqui/results/vits/wass_all/vits_wass-de-November-18-2025_09+07AM-c0749fb6/config.json"

from TTS.utils.synthesizer import Synthesizer

synthesizer = Synthesizer(
    tts_checkpoint=model_path,
    tts_config_path=config_path,
)

wav = synthesizer.tts(text)  # Audio erzeugen
synthesizer.save_wav(wav, "speech_out.wav")
