from curses import meta
import os
from pydoc import text

from numpy import character
from sympy import use

# from TTS.tts.utils.text import characters
from trainer import Trainer, TrainerArgs

from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import CharactersConfig, Vits, VitsArgs, VitsAudioConfig
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.text.tokenizer_phoneme_input2 import TTSTokenizer
from TTS.utils.audio import AudioProcessor

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")

speakers = ["hpo"] # vd and at
    
output_path = "/home/projects/vokquant/coqui/results/vits/wass_"+"_".join(speakers)+"_phoneme_input" # adds _ between additional speaker names

def main():
    dataset_config = BaseDatasetConfig(
        formatter="wass",
        # meta_file_train="train-text-pac_at_vd.txt",
        meta_file_train="single_speaker_metadata/train-text_"+ "_".join(speakers) + "_phonemes_space.txt",
        path="/home/projects/vokquant/data/aridialect/"
    )

    audio_config = VitsAudioConfig(
        sample_rate=22050, win_length=1024, hop_length=256, num_mels=80, mel_fmin=0, mel_fmax=None
    )

    vitsArgs = VitsArgs(
        use_speaker_embedding=True,
    )
    
    use_phonemes = True
    if use_phonemes:
        # text_cleaner = "multilingual_phoneme_cleaners"
        text_cleaner = None
    else:
        text_cleaner = "multilingual_cleaners"
        
    config = VitsConfig(
        model_args=vitsArgs,
        audio=audio_config,
        run_name="vits_wass-de",
        batch_size=32,
        eval_batch_size=16,
        batch_group_size=5,
        num_loader_workers=4,
        num_eval_loader_workers=4,
        run_eval=True,
        test_delay_epochs=1,
        epochs=1000,
        text_cleaner=None,
        use_phonemes=True,
        phoneme_language="de",
        phonemizer=None,
        phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
        compute_input_seq_cache=True,
        print_step=1000,
        print_eval=False,
        mixed_precision=True,
        characters=CharactersConfig(
            characters_class="TTS.tts.models.vits.VitsPhonemeCharacters",  # Update this path
            pad="<PAD>",
            blank="<BLNK>",
            punctuations="",  # Add punctuations if needed: "!?.,"
            characters=[],  # No additional characters needed for phoneme input
            phonemes=[  # ONLY these phonemes will be used
                "a", "aː", "b", "d", "e", "eː", "f", "h", "i", "iː", "j", "k", "l", "m", "n",
                "o", "oː", "p", "r", "s", "sː", "t", "u", "uː", "v", "x", "y", "z",
                "~", "ç", "ð", "ø", "øː", "ŋ", "œ", "œː", "ɐ", "ɐː", "ɑ", "ɑː", "ɒː", "ɔ", "ɔː",
                "ə", "ɛ", "ɛː", "ɡ", "ɣ", "ɪ", "ɶ", "ʀ", "ʃ", "ʊ", "ʊː", "ʎ", "ʏ", "ʏː", "ʒ", "ʔ", "β", "χ"
            ]
        ),
        test_sentences=[
            "v ɛː ɐ t r ɪ ŋ k t ʔ a ɪ n ə n k a f eː",
            "z iː z ɔ l t ə m eː d iː t s iː n eː m ə n"
        ],
        max_text_len=325,
        output_path=output_path,
        datasets=[dataset_config],
        cudnn_benchmark=False,
    )
    # INITIALIZE THE AUDIO PROCESSOR
    # Audio processor is used for feature extraction and audio I/O.
    # It mainly serves to the dataloader and the training loggers.
    ap = AudioProcessor.init_from_config(config)

    # INITIALIZE THE TOKENIZER
    # Tokenizer is used to convert text to sequences of token IDs.
    # config is updated with the default characters if not defined in the config.
    tokenizer, config = TTSTokenizer.init_from_config(config)
    
    # Check vocab size
    chars = config.characters
    # print(f"Total vocab size: {len(chars)}")
    # print("Chars:", chars)
    # print(f"Vocab: {chars}")
    # print(f"Expected: {1 + 0 + 63 + 1} = 65")
    # LOAD DATA SAMPLES
    # Each sample is a list of ```[text, audio_file_path, speaker_name]```
    # You can define your custom sample loader returning the list of samples.
    # Or define your custom formatter and pass it to the `load_tts_samples`.
    # Check `TTS.tts.datasets.load_tts_samples` for more details.
    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    # init speaker manager for multi-speaker training
    # it maps speaker-id to speaker-name in the model and data-loader
    speaker_manager = SpeakerManager()
    speaker_manager.set_ids_from_data(train_samples + eval_samples, parse_key="speaker_name")
    config.model_args.num_speakers = speaker_manager.num_speakers

    # init model
    model = Vits(config, ap, tokenizer, speaker_manager)

    # init the trainer and 🚀
    trainer = Trainer(
        TrainerArgs(),
        config,
        output_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
