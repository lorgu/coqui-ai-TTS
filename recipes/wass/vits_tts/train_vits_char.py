import os

from trainer import Trainer, TrainerArgs

from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import CharactersConfig, Vits, VitsAudioConfig
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

speaker = "hpo"
output_path = "/home/projects/vokquant/coqui/results/vits/wass" + "_" + speaker + "_single_char"

def main():
    dataset_config = BaseDatasetConfig(
        formatter="wass",
        # meta_file_train="train-text-pac_orig.txt",
        meta_file_train="single_speaker_metadata/train-text_" + speaker + ".txt",
        path="/home/projects/vokquant/data/aridialect/"
    )

    # download dataset if not already present
    if not os.path.exists(dataset_config.path):
        print("Dataset not found.")

    audio_config = VitsAudioConfig(
        sample_rate=22050,
        win_length=1024,
        hop_length=256,
        num_mels=80,
        mel_fmin=0,
        mel_fmax=None,
    )

    config = VitsConfig(
        audio=audio_config,
        run_name="vits_wass-de",
        batch_size=32,
        eval_batch_size=16,
        batch_group_size=5,
        num_loader_workers=4,
        num_eval_loader_workers=4,
        run_eval=True,
        test_delay_epochs=-1,
        epochs=1000,
        text_cleaner="multilingual_cleaners",
        use_phonemes=False,
        phoneme_language="de",
        phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
        compute_input_seq_cache=True,
        print_step=25,
        print_eval=True,
        mixed_precision=True,
        characters=CharactersConfig(
            characters_class="TTS.tts.models.vits.VitsCharacters",
            pad="<PAD>",
            eos="<EOS>",
            bos="<BOS>",
            blank="<BLNK>",
            characters="!',-.:?_ abcdefghijklmnopqrstuvwxyzÄÖÜßäöü",
            punctuations="!',-.:?_ ",
            phonemes=None,
        ),
        test_sentences=[
            "Es hat mich viel Zeit gekostet eine Stimme zu entwickeln, jetzt wo ich sie habe werde ich nicht mehr schweigen.",
            "Es tut mir Leid David. Das kann ich leider nicht machen.",
            "Dieser Kuchen ist großartig. Er ist so lecker und feucht.",
            "Vor dem 22. November 1963.",
            "schau amoe rüba zu ia, woont e glei in da Nee",
            "wos maanst, waun ma den koch amoe an besuch obschdotn?",
            "Warum ist das in dem Sektor so wichtig?",
            "Dawäu bleib I no do",
            "Gema hakln!"
        ],
        output_path=output_path,
        datasets=[dataset_config],
    )
    
    # force the convertion of the custom characters to a config attribute
    config.from_dict(config.to_dict())

    # INITIALIZE THE AUDIO PROCESSOR
    # Audio processor is used for feature extraction and audio I/O.
    # It mainly serves to the dataloader and the training loggers.
    ap = AudioProcessor.init_from_config(config)

    # INITIALIZE THE TOKENIZER
    # Tokenizer is used to convert text to sequences of token IDs.
    # config is updated with the default characters if not defined in the config.
    tokenizer, config = TTSTokenizer.init_from_config(config)

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

    # init model
    model = Vits(config, ap, tokenizer, speaker_manager=None)

    # init the trainer and 🚀
    trainer = Trainer(
        TrainerArgs(),
        config,
        output_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
        # callbacks={"on_epoch_end": post_epoch_hook},
    )
    trainer.fit()


if __name__ == "__main__":
    main()
