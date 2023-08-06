import os
from pathlib import Path
from tqdm import tqdm
import json
from typing import Callable, Dict, List, Optional, Union
from hydra.utils import instantiate
from nemo.core.config import hydra_runner


import torch
import torchaudio
from transformers import AutoTokenizer, AutoModel, AutoModelForAudioXVector

from nemo.collections.tts.torch.data import TTSDataset

from nemo.collections.asr.parts.preprocessing.features import WaveformFeaturizer
from nemo.collections.asr.parts.preprocessing.segment import AudioSegment
from nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers import (
    BaseTokenizer,
    EnglishCharsTokenizer,
    EnglishPhonemesTokenizer,
)
from nemo.collections.tts.torch.helpers import (
    BetaBinomialInterpolator,
    beta_binomial_prior_distribution,
    general_padding,
    get_base_dir,
)
from nemo.collections.tts.torch.tts_data_types import (
    DATA_STR2DATA_CLASS,
    MAIN_DATA_TYPES,
    AlignPriorMatrix,
    Durations,
    Energy,
    LMTokens,
    LogMel,
    P_voiced,
    Pitch,
    SpeakerID,
    TTSDataType,
    Voiced_mask,
    WithLens,
)
from nemo.core.classes import Dataset
from nemo.utils import logging

try:
    from nemo_text_processing.text_normalization.normalize import Normalizer

    PYNINI_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    Normalizer = None
    PYNINI_AVAILABLE = False

EPSILON = 1e-9
WINDOW_FN_SUPPORTED = {
    'hann': torch.hann_window,
    'hamming': torch.hamming_window,
    'blackman': torch.blackman_window,
    'bartlett': torch.bartlett_window,
    'none': None,
}


class MultiTTSDataset(TTSDataset):
    def __init__(
            self,
            manifest_filepath: Union[str, Path, List[str], List[Path]],
            sample_rate: int,
            text_tokenizer: Union[BaseTokenizer, Callable[[str], List[int]]],
            tokens: Optional[List[str]] = None,
            text_normalizer: Optional[Union[Normalizer, Callable[[str], str]]] = None,
            text_normalizer_call_kwargs: Optional[Dict] = None,
            text_tokenizer_pad_id: Optional[int] = None,
            sup_data_types: Optional[List[str]] = None,
            sup_data_path: Optional[Union[Path, str]] = None,
            max_duration: Optional[float] = None,
            min_duration: Optional[float] = None,
            ignore_file: Optional[Union[str, Path]] = None,
            trim: bool = False,
            trim_ref: Optional[float] = None,
            trim_top_db: Optional[int] = None,
            trim_frame_length: Optional[int] = None,
            trim_hop_length: Optional[int] = None,
            n_fft: int = 1024,
            win_length: Optional[int] = None,
            hop_length: Optional[int] = None,
            window: str = "hann",
            n_mels: int = 80,
            lowfreq: int = 0,
            highfreq: Optional[int] = None,
            use_spk_id: bool = False,
            use_xvector: bool = False,
            **kwargs,
    ):
        if use_spk_id==True and use_xvector==True:
            raise ValueError("You either use speaker id or xvector for speaker embedding, you cannot use both")

        super().__init__(manifest_filepath=manifest_filepath,
                         sample_rate=sample_rate,
                         text_tokenizer=text_tokenizer,
                         text_normalizer=text_normalizer,
                         text_normalizer_call_kwargs=text_normalizer_call_kwargs,
                         sup_data_path=sup_data_path,
                         sup_data_types=sup_data_types,
                         n_fft=n_fft,
                         win_length=win_length,
                         hop_length=hop_length,
                         window=window,
                         n_mels=n_mels,
                         lowfreq=lowfreq,
                         highfreq=highfreq,
                         max_duration=max_duration,
                         min_duration=min_duration,
                         ignore_file=ignore_file,
                         trim=trim,
                         trim_top_db=trim_top_db,
                         **kwargs)

        lm_model = kwargs.pop("lm_model")
        audio_model = kwargs.pop("audio_model")
        self.use_xvector = use_xvector
        self.use_spk_id = use_spk_id

        self.pretrained_model_name = lm_model.split("/")[-1].split("-")[0] + "_" + \
                           audio_model.split("/")[-1].split('-')[0]
        self.pretrained_model_name = Path(self.sup_data_path) / self.pretrained_model_name
        if not os.path.exists(self.pretrained_model_name):
            os.makedirs(self.pretrained_model_name, exist_ok=True)

        self.lm_model_tokenizer = AutoTokenizer.from_pretrained(lm_model)
        self.lm_model = AutoModel.from_pretrained(lm_model).eval()
        self.audio_model = AutoModel.from_pretrained(audio_model).eval()
        if self.use_xvector:
            xvector_model = kwargs.pop("xvector_model")
            self.xvector_model = AutoModelForAudioXVector.from_pretrained(xvector_model).eval()

        lm_folder = Path(self.pretrained_model_name) / "lm_embedding"
        if not os.path.exists(lm_folder):
            os.makedirs(lm_folder)

        audio_folder = Path(self.pretrained_model_name) / "audio_embedding"
        if not os.path.exists(audio_folder):
            os.makedirs(audio_folder)

        if self.use_xvector:
            speaker_folder = Path(self.pretrained_model_name) / "speaker_embedding"
            if not os.path.exists(speaker_folder):
                os.makedirs(speaker_folder)

    def __getitem__(self, index):
        try:
            (
                audio,
                audio_length,
                text,
                text_length,
                log_mel,
                log_mel_length,
                durations,
                align_prior_matrix,
                pitch,
                pitch_length,
                energy,
                energy_length,
                speaker_id,
                voiced_mask,
                p_voiced,
            ) = super().__getitem__(index)
        except AttributeError:
            print(self.data[index]['audio_filepath'])
        sample = self.data[index]
        # find lm, audio embedding from sup_data path
        # if it exists, load if
        # else, run model and save it


        lm_path = self.pretrained_model_name / "lm_embedding"
        audio_path = self.pretrained_model_name / "audio_embedding"
        speaker_path = self.pretrained_model_name / "speaker_embedding"
        file_name = sample['audio_filepath'].split("/")[-1]
        file_name = file_name.replace("wav", "pt")

        lm_file = os.path.join(lm_path, file_name)
        if os.path.exists(lm_file):
            try:
                lm_embedding = torch.load(lm_file)
            except RuntimeError:
                lm_inputs = self.lm_model_tokenizer(sample['original_text'], return_tensors='pt')
                lm_embedding = self.lm_model(**lm_inputs).last_hidden_state
                lm_embedding = lm_embedding.squeeze(0)
                torch.save(lm_embedding, lm_file)
        else:
            lm_inputs = self.lm_model_tokenizer(sample['original_text'], return_tensors='pt')
            lm_embedding = self.lm_model(**lm_inputs).last_hidden_state
            lm_embedding = lm_embedding.squeeze(0)
            torch.save(lm_embedding, lm_file)

        audio_file = os.path.join(audio_path, file_name)
        if os.path.exists(audio_file):
            try:
                audio_embedding = torch.load(audio_file)
            except RuntimeError:
                audio_input, sr = torchaudio.load(sample['audio_filepath'])
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
                audio_input = resampler(audio_input)
                audio_embedding = self.audio_model(input_values=audio_input).last_hidden_state
                audio_embedding = audio_embedding.squeeze(0)
                torch.save(audio_embedding, audio_file)
        else:
            audio_input, sr = torchaudio.load(sample['audio_filepath'])
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            audio_input = resampler(audio_input)
            audio_embedding = self.audio_model(input_values=audio_input).last_hidden_state
            audio_embedding = audio_embedding.squeeze(0)
            torch.save(audio_embedding, audio_file)

        speaker_embedding = None
        if self.use_xvector:
            speaker_file = os.path.join(speaker_path, file_name)
            if os.path.exists(speaker_file):
                speaker_embedding = torch.load(speaker_file)
            else:
                audio_input, sr = torchaudio.load(sample['audio_filepath'])
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
                audio_input = resampler(audio_input)
                speaker_embedding = self.xvector_model(input_values=audio_input).embeddings
                speaker_embedding = speaker_embedding.squeeze(0)
                torch.save(speaker_embedding, speaker_file)
        if speaker_embedding != None:
            speaker_embedding = speaker_embedding.detach()

        audio_embedding_length = torch.tensor(audio_embedding.size(0)).long()
        lm_embedding_length = torch.tensor(lm_embedding.size(0)).long()

        return (
            audio,
            audio_length,
            text,
            text_length,
            log_mel,
            log_mel_length,
            durations,
            align_prior_matrix,
            pitch,
            pitch_length,
            energy,
            energy_length,
            speaker_id,
            voiced_mask,
            p_voiced,
            audio_embedding,
            audio_embedding_length,
            lm_embedding,
            lm_embedding_length,
            speaker_embedding,
        )

    def general_collate_fn(self, batch):
        (
            _,
            audio_lengths,
            _,
            tokens_lengths,
            _,
            log_mel_lengths,
            durations_list,
            align_prior_matrices_list,
            pitches,
            pitches_lengths,
            energies,
            energies_lengths,
            _,
            voiced_masks,
            p_voiceds,
            audio_embeddings,
            audio_embedding_lengths,
            lm_embeddings,
            lm_embedding_lengths,
            speaker_embeddings,
        ) = zip(*batch)

        max_audio_len = max(audio_lengths).item()
        max_tokens_len = max(tokens_lengths).item()
        max_log_mel_len = max(log_mel_lengths) if LogMel in self.sup_data_types_set else None
        max_durations_len = max([len(i) for i in durations_list]) if Durations in self.sup_data_types_set else None
        max_pitches_len = max(pitches_lengths).item() if Pitch in self.sup_data_types_set else None
        max_energies_len = max(energies_lengths).item() if Energy in self.sup_data_types_set else None
        max_audio_embedding_len = max(audio_embedding_lengths).item()
        max_lm_embedding_len = max(lm_embedding_lengths).item()

        if LogMel in self.sup_data_types_set:
            log_mel_pad = torch.finfo(batch[0][4].dtype).tiny

        align_prior_matrices = (
            torch.zeros(
                len(align_prior_matrices_list),
                max([prior_i.shape[0] for prior_i in align_prior_matrices_list]),
                max([prior_i.shape[1] for prior_i in align_prior_matrices_list]),
            )
            if AlignPriorMatrix in self.sup_data_types_set
            else []
        )
        audios, tokens, log_mels, durations_list, pitches, energies, speaker_ids, voiced_masks, p_voiceds, audio_embeddings, lm_embeddings = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            []
        )

        for i, sample_tuple in enumerate(batch):
            (
                audio,
                audio_len,
                token,
                token_len,
                log_mel,
                log_mel_len,
                durations,
                align_prior_matrix,
                pitch,
                pitch_length,
                energy,
                energy_length,
                speaker_id,
                voiced_mask,
                p_voiced,
                audio_embedding,
                audio_embedding_length,
                lm_embedding,
                lm_embedding_length,
                speaker_embedding,
            ) = sample_tuple

            audio = general_padding(audio, audio_len.item(), max_audio_len)
            audios.append(audio)

            token = general_padding(token, token_len.item(), max_tokens_len, pad_value=self.text_tokenizer_pad_id)
            tokens.append(token)

            if LogMel in self.sup_data_types_set:
                log_mels.append(general_padding(log_mel, log_mel_len, max_log_mel_len, pad_value=log_mel_pad))

            if Durations in self.sup_data_types_set:
                durations_list.append(general_padding(durations, len(durations), max_durations_len))

            if AlignPriorMatrix in self.sup_data_types_set:
                align_prior_matrices[
                i, : align_prior_matrix.shape[0], : align_prior_matrix.shape[1]
                ] = align_prior_matrix

            if Pitch in self.sup_data_types_set:
                pitches.append(general_padding(pitch, pitch_length.item(), max_pitches_len))

            if Voiced_mask in self.sup_data_types_set:
                voiced_masks.append(general_padding(voiced_mask, pitch_length.item(), max_pitches_len))

            if P_voiced in self.sup_data_types_set:
                p_voiceds.append(general_padding(p_voiced, pitch_length.item(), max_pitches_len))

            if Energy in self.sup_data_types_set:
                energies.append(general_padding(energy, energy_length.item(), max_energies_len))

            if SpeakerID in self.sup_data_types_set:
                speaker_ids.append(speaker_id)

            pad = torch.zeros(max_audio_embedding_len - audio_embedding_length.item(), audio_embedding.size(-1))
            audio_embedding = torch.cat([audio_embedding, pad], dim=0)
            audio_embeddings.append(audio_embedding.detach())

            pad = torch.zeros(max_lm_embedding_len - lm_embedding_length.item(), lm_embedding.size(-1))
            lm_embedding = torch.cat([lm_embedding, pad], dim=0)
            lm_embeddings.append(lm_embedding.detach())

        data_dict = {
            "audio": torch.stack(audios),
            "audio_lens": torch.stack(audio_lengths),
            "text": torch.stack(tokens),
            "text_lens": torch.stack(tokens_lengths),
            "log_mel": torch.stack(log_mels) if LogMel in self.sup_data_types_set else None,
            "log_mel_lens": torch.stack(log_mel_lengths) if LogMel in self.sup_data_types_set else None,
            "durations": torch.stack(durations_list) if Durations in self.sup_data_types_set else None,
            "align_prior_matrix": align_prior_matrices if AlignPriorMatrix in self.sup_data_types_set else None,
            "pitch": torch.stack(pitches) if Pitch in self.sup_data_types_set else None,
            "pitch_lens": torch.stack(pitches_lengths) if Pitch in self.sup_data_types_set else None,
            "energy": torch.stack(energies) if Energy in self.sup_data_types_set else None,
            "energy_lens": torch.stack(energies_lengths) if Energy in self.sup_data_types_set else None,
            "speaker_id": torch.stack(speaker_ids) if SpeakerID in self.sup_data_types_set else None,
            "voiced_mask": torch.stack(voiced_masks) if Voiced_mask in self.sup_data_types_set else None,
            "p_voiced": torch.stack(p_voiceds) if P_voiced in self.sup_data_types_set else None,
            "audio_embedding": torch.stack(audio_embeddings),
            "audio_embedding_lens": torch.stack(audio_embedding_lengths),
            "lm_embedding": torch.stack(lm_embeddings),
            "lm_embedding_lens": torch.stack(lm_embedding_lengths),
            "speaker_embedding": torch.stack(speaker_embeddings) if self.use_xvector else None
        }
        return data_dict

    def join_data(self, data_dict):
        result = []
        for data_type in MAIN_DATA_TYPES + self.sup_data_types:
            result.append(data_dict[data_type.name])

            if issubclass(data_type, TTSDataType) and issubclass(data_type, WithLens):
                result.append(data_dict[f"{data_type.name}_lens"])
        result.append(data_dict['audio_embedding'])
        result.append(data_dict['audio_embedding_lens'])
        result.append(data_dict['lm_embedding'])
        result.append(data_dict['lm_embedding_lens'])
        if data_dict['speaker_embedding'] != None:
            result.append(data_dict['speaker_embedding'])
        return tuple(result)

from torch.utils.data.distributed import DistributedSampler


class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    """
    Maintain similar input lengths in a batch.
    Length groups are specified by boundaries.
    Ex) boundaries = [b1, b2, b3] -> any batch is included either {x | b1 < length(x) <=b2} or {x | b2 < length(x) <= b3}.

    It removes samples which are not included in the boundaries.
    Ex) boundaries = [b1, b2, b3] -> any x s.t. length(x) <= b1 or length(x) > b3 are discarded.
    """

    def __init__(self, dataset, batch_size, boundaries, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries

        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas

    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)

        for i in range(len(buckets) - 1, 0, -1):
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i + 1)

        num_samples_per_bucket = []
        total_batch_size = self.num_replicas * self.batch_size
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            rem = (total_batch_size - (len_bucket % total_batch_size)) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = []
        if self.shuffle:
            for bucket in self.buckets:
                indices.append(torch.randperm(len(bucket), generator=g).tolist())
        else:
            for bucket in self.buckets:
                indices.append(list(range(len(bucket))))

        batches = []
        for i in range(len(self.buckets)):
            bucket = self.buckets[i]
            len_bucket = len(bucket)
            ids_bucket = indices[i]
            num_samples_bucket = self.num_samples_per_bucket[i]

            # add extra samples to make it evenly divisible
            rem = num_samples_bucket - len_bucket
            ids_bucket = ids_bucket + ids_bucket * (rem // len_bucket) + ids_bucket[: (rem % len_bucket)]

            # subsample
            ids_bucket = ids_bucket[self.rank:: self.num_replicas]

            # batching
            for j in range(len(ids_bucket) // self.batch_size):
                batch = [bucket[idx] for idx in ids_bucket[j * self.batch_size: (j + 1) * self.batch_size]]
                batches.append(batch)

        if self.shuffle:
            batch_ids = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in batch_ids]
        self.batches = batches

        assert len(self.batches) * self.batch_size == self.num_samples
        return iter(self.batches)

    def _bisect(self, x, lo=0, hi=None):
        if hi is None:
            hi = len(self.boundaries) - 1

        if hi > lo:
            mid = (hi + lo) // 2
            if self.boundaries[mid] < x and x <= self.boundaries[mid + 1]:
                return mid
            elif x <= self.boundaries[mid]:
                return self._bisect(x, lo, mid)
            else:
                return self._bisect(x, mid + 1, hi)
        else:
            return -1

    def __len__(self):
        return self.num_samples // self.batch_size

    def set_epoch(self, epoch: int) -> None:
        """
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.
        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


@hydra_runner(config_path="../conf/english", config_name='ds_for_data2vec_vctk.yaml')
def main(cfg):
    torch.multiprocessing.set_sharing_strategy('file_system')
    dataset = instantiate(cfg.dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset= dataset, batch_size=1, collate_fn=dataset._collate_fn, num_workers=0
    )

    pitch_list = []
    for batch in tqdm(dataloader, total=len(dataloader)):
        audios, audio_lengths, tokens, token_lengths, attn_prior, pitches, pitches_lengths, energy, energy_lens, spk_id, audio_embeddings, audio_emb_length, lm_embeddings, lm_emb_length = batch
        pitch = pitches.squeeze(0)
        pitch_list.append(pitch[pitch != 0])

    pitch_tensor = torch.cat(pitch_list)
    pitch_mean, pitch_std = pitch_tensor.mean().item(), pitch_tensor.std().item()
    pitch_min, pitch_max = pitch_tensor.min().item(), pitch_tensor.max().item()
    print(f"PITCH_MEAN={pitch_mean}, PITCH_STD={pitch_std}")
    print(f"PITCH_MIN={pitch_min}, PITCH_MAX={pitch_max}")
    f = open(os.path.join(cfg.sup_data_path, "pitch_stats.txt"), 'w')
    f.write(f"PITCH MEAN : {pitch_mean}\n")
    f.write(f"PITCH STD : {pitch_std}\n")
    f.write(f"PITCH MIN : {pitch_min}\n")
    f.write(f"PITCH MAX : {pitch_max}\n")
    f.close()

if __name__ == "__main__":
    main()