import os

from trainer import Trainer, TrainerArgs

from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import CharactersConfig, Vits, VitsAudioConfig
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
# as in checkpoint config:
from TTS.tts.utils.text.characters import IPAPhonemes

speaker = "hpo"
output_path = "/home/projects/vokquant/coqui/results/vits/wass" + "_" + speaker + "_single"

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
        num_loader_workers=0,
        num_eval_loader_workers=4,
        run_eval=True,
        test_delay_epochs=-1,
        epochs=1000,
        text_cleaner="multilingual_phoneme_cleaners",
        use_phonemes=True,
        phonemizer="gruut", # as in config of checkpoint
        # phoneme_language="de",
        phoneme_language="de-de", # as in config of checkpoint
        characters=CharactersConfig(
            characters_class="TTS.tts.utils.text.characters.IPAPhonemes",  # string, not class
            vocab_dict=None,
            pad="<PAD>",
            eos="<EOS>",
            bos="<BOS>",
            blank="<BLNK>",
            characters="iy\u0268\u0289\u026fu\u026a\u028f\u028ae\u00f8\u0258\u0259\u0275\u0264o\u025b\u0153\u025c\u025e\u028c\u0254\u00e6\u0250a\u0276\u0251\u0252\u1d7b\u0298\u0253\u01c0\u0257\u01c3\u0284\u01c2\u0260\u01c1\u029bpbtd\u0288\u0256c\u025fk\u0261q\u0262\u0294\u0274\u014b\u0272\u0273n\u0271m\u0299r\u0280\u2c71\u027e\u027d\u0278\u03b2fv\u03b8\u00f0sz\u0283\u0292\u0282\u0290\u00e7\u029dx\u0263\u03c7\u0281\u0127\u0295h\u0266\u026c\u026e\u028b\u0279\u027bj\u0270l\u026d\u028e\u029f\u02c8\u02cc\u02d0\u02d1\u028dw\u0265\u029c\u02a2\u02a1\u0255\u0291\u027a\u0267\u02b2\u025a\u02de\u026b",
            punctuations="!'(),-.:;? ",
            phonemes=None,
            is_unique=False,
            is_sorted=True
        ),
        add_blank=True,
        phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
        compute_input_seq_cache=True,
        print_step=25,
        print_eval=True,
        mixed_precision=True,
        test_sentences=[
            "Es hat mich viel Zeit gekostet ein Stimme zu entwickeln, jetzt wo ich sie habe werde ich nicht mehr schweigen.",
            "Sei eine Stimme, kein Echo.",
            "Es tut mir Leid David. Das kann ich leider nicht machen.",
            "Dieser Kuchen ist großartig. Er ist so lecker und feucht.",
            "Vor dem 22. November 1963.",
        ],
        output_path=output_path,
        datasets=[dataset_config],
    )

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
