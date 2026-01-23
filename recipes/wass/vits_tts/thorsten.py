import torch
from TTS.api import TTS

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# List available 🐸TTS models
# print(TTS().list_models())

# Initialize TTS
tts = TTS("tts_models/de/thorsten/vits").to(device)

# List speakers
# print(tts.speakers)

# Run TTS
# ❗ XTTS supports both, but many models allow only one of the `speaker` and
# `speaker_wav` arguments

# TTS with list of amplitude values as output, clone the voice from `speaker_wav`
# wav = tts.tts(
#   text="Hello world!",
#   speaker_wav="my/cloning/audio.wav",
#   language="en"
# )

# TTS to a file, use a preset speaker
tts.tts_to_file(
  text="Huhu. Der Tag ist vorüber und die Nacht bricht herein. Trete ein, hübsches Kotelett. In das Reich der koteks. ",
  speaker_wav="/home/projects/vokquant/data/aridialect/aridialect_wav_22050/alf_at_berlin_001.wav",
#   language="de",
  file_path="/home/projects/vokquant/coqui/results/vits/wass_spo/vits_wass-de-November-16-2025_01+39PM-c0749fb6/wav/output_thorsten.wav"
)