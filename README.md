# Phoneme Level Lyrics Aligner

This repository can be used to align lyrics transcripts with the corresponding audio signals. The audio signals may contain solo singing or singing voice mixed with other instruments.
It contains a trained deep neural network which performs alignment and singing voice separation jointly.

Details about the model, training, and data are described in the associated paper:
> Schulze-Forster, K., Doire, C., Richard, G., & Badeau, R.
> _"Phoneme Level Lyrics Alignment and Text-Informed Singing Voice Separation."_
> IEEE/ACM Transactions on Audio, Speech and Language Processing (2021).
> DOI: [10.1109/TASLP.2021.3091817](https://doi.org/10.1109/TASLP.2021.3091817).
> Public version [available here](https://hal.telecom-paris.fr/hal-03255334/file/2021_Phoneme_level_lyrics_alignment_and_text-informed_singing_voice_separation.pdf).

If you use the model or code, please cite the paper:

<pre>
@article{schulze2021phoneme,
    author={Schulze-Forster, Kilian and Doire, Clement S. J. and Richard, Gaël and Badeau, Roland},
    journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
    title={Phoneme Level Lyrics Alignment and Text-Informed Singing Voice Separation},
    year={2021},
    volume={29},
    number={},
    pages={2382-2395},
    doi={10.1109/TASLP.2021.3091817}
}
</pre>

---

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Caio99BR/lyrics-aligner.git
    cd lyrics-aligner
    ```

2. Install the required packages with pip:
    ```bash
    pip install pyqt5 decorator ffmpeg audioread resampy librosa pysoundfile praatio torchvision torchaudio paramiko cryptography pyopenssl
    ```

---

## Data Preparation

### Audio
Place all audio files in `input` directory (You can use sub-folders). Audio files are loaded using `librosa`, so all formats supported by `librosa` (e.g., `.wav`, `.mp3`) are accepted. See the [librosa documentation](https://librosa.org/doc/latest/index.html) for more.

### Lyrics
Place all lyrics files in the `input` directory (You can use sub-folders). Each `.txt` lyrics file must have the same name as the corresponding audio file (e.g., `input/song1.wav` ➝ `input/song1.txt`).

You can provide lyrics as words or as phonemes.

#### Phoneme Format
- Use only the 39 ARPAbet phonemes listed [here](http://www.speech.cs.cmu.edu/cgi-bin/cmudict).
- One phoneme per line.
- The first and last symbols should be a space character `>`.
- Use `>` between words or wherever silence is expected.

> Note: If lyrics are given as phonemes, only phoneme onsets will be computed.

#### Word Format
If providing lyrics as words:

1. Create a list of unique words:
    ```bash
    python make_word_list.py PATH/TO/LYRICS --dataset-name NAME
    ```

2. Go to [CMU LexTool](http://www.speech.cs.cmu.edu/tools/lextool.html) and upload `NAME_word_list.txt`.

3. Copy the generated `.dict` file content and paste it into `input/NAME_word2phoneme.txt`.

4. Convert it into a phoneme dictionary:
    ```bash
    python make_word2phoneme_dict.py --dataset-name NAME
    ```

---

## Usage
To compute phoneme and/or word onsets:
    ```bash
    python align.py PATH/TO/INPUTS --lyrics-format w --onsets p --dataset-name dataset1 --vad-threshold 0
    ```

---

## Optional Flags

- `--lyrics-format`
  Must be `w` if the lyrics are provided as words (and have been processed as described above) and `p` if the lyrics are provided as phonemes.

- `--onsets`
  If phoneme onsets should be computed, set to `p`. If word onsets should be computed, set to `w`. If phoneme and word onsets should be computed, set to `pw` (only possible if lyrics are provided as words).

- `--dataset-name`
  Should be the same as used for data preparation above.

- `--vad-threshold`
  The model also computes an estimate of the isolated singing voice which can be used as Voice Activity Detector (VAD). This may be useful in challenging scenarios where long pauses are made by the singer while instruments are playing (e.g., intro, soli, outro). The magnitude of the vocals estimate is computed. Here a threshold (float) can be set to discriminate between active and inactive voice given the magnitude. The default is 0, which means that no VAD is used. The optimal value for a given audio signal may be difficult to determine as it depends on the loudness of the voice. In our experiments, we used values between 0 and 30. You could print or plot the voice magnitude (computed in line 235) to get an intuition for an appropriate value. We recommend using the option only if large errors are made on audio files with long instrumental sections.

- `PATH/TO/INPUTS`
  Specifies the directory containing the input audio and lyrics files. By default, the script looks for files in the `inputs` directory. If you wish to use a different directory, provide its path.

---

## Acknowledgment

This project received funding from the European Union's Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement No. 765068.

---

## Copyright

© 2021 Kilian Schulze-Forster, Télécom Paris, Institut Polytechnique de Paris. All rights reserved.
