# dc-comix-tts
Implementation of [DCComix TTS: An End-to-End Expressive TTS with Discrete Code Collaborated with Mixer](https://arxiv.org/abs/2305.19567)
Accepted to Interspech 2023. Audio samples/demo for this system is [here](https://lakahaga.github.io/dc-comix-tts/)

Abstract: Despite the huge successes made in neutral TTS, content-leakage remains a challenge. In this paper, we propose a new input representation and simple architecture to achieve improved prosody modeling. Inspired by the recent success in the use of discrete code in TTS, we introduce discrete code to the input of the reference encoder. Specifically, we leverage the vector quantizer from the audio compression model to exploit the diverse acoustic information it has already been trained on. In addition, we apply the modified MLP-Mixer to the reference encoder, making the architecture lighter. As a result, we train the prosody transfer TTS in an end-to-end manner. We prove the effectiveness of our method through both subjective and objective evaluations. We demonstrate that the reference encoder learns better speaker-independent prosody when discrete code is utilized as input in the experiments. In addition, we obtain comparable results even when fewer parameters are inputted.

* This repository leverages [Nemo](https://github.com/NVIDIA/NeMo) for [VITS](https://arxiv.org/pdf/2106.06103.pdf) and [MixerTTS](https://arxiv.org/abs/2110.03584) implementation.
* We use [Encodec](https://github.com/facebookresearch/encodec) for discrete code

## Installation
 * python ≥ 3.8
 * pytorch 1.11.0+cu113
 * nemo_toolkit 1.18.0
 
 See `requirements.txt` for other libraries
## Traininig
* prepare data ([VCTK](https://datashare.ed.ac.uk/handle/10283/2651))
  ```
  python preprocess/make_manifest.py
    ```
  * Note that we resample VCTK audios to 24kHz to match resolution with Encodec
* preprocessing
  * text normalization
  ```
  python torchdata/text_preprocess.py
  ```
* run `train.py`
  * for `dc-comix-tts` : use `ref_mixer_codec_vits.yaml`
  
## References

```text
@software{Harper_NeMo_a_toolkit,
author = {Harper, Eric and Majumdar, Somshubra and Kuchaiev, Oleksii and Jason, Li and Zhang, Yang and Bakhturina, Evelina and Noroozi, Vahid and Subramanian, Sandeep and Nithin, Koluguri and Jocelyn, Huang and Jia, Fei and Balam, Jagadeesh and Yang, Xuesong and Livne, Micha and Dong, Yi and Naren, Sean and Ginsburg, Boris},
title = {{NeMo: a toolkit for Conversational AI and Large Language Models}},
url = {https://github.com/NVIDIA/NeMo}
}
```
```text
@article{defossez2022highfi,
  title={High Fidelity Neural Audio Compression},
  author={Défossez, Alexandre and Copet, Jade and Synnaeve, Gabriel and Adi, Yossi},
  journal={arXiv preprint arXiv:2210.13438},
  year={2022}
}
```
