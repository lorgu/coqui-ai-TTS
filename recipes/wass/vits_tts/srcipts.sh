# uses tts_models/de/thorsten/vits as checkpoint for finetuning

CUDA_VISIBLE_DEVICES="0" python recipes/wass/vits_tts/train_vits.py \
    --config_path ~/.local/share/tts/tts_models--de--thorsten--vits/config.json \
    --restore_path  ~/.local/share/tts/tts_models--de--thorsten--vits/model.pth
    


# or
tts --list_models

