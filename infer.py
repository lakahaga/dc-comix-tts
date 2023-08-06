from argparse import ArgumentParser
from pathlib import Path
import json
import os
from copy import deepcopy
from tqdm import tqdm
import time
from statistics import mean

import soundfile as sf
import torch
import torchaudio
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from model.vits import VitsModel
from module.ref_gst import GlobalStyleTokenForCodec

from nemo.core.config import hydra_runner
from hydra.utils import instantiate

def match_file_for_transfer(manifests):
    # 108 speakers
    file_ids = [x['audio_filepath'] for x in manifests]
    file_ids = [x.split("/")[-1].replace(".wav", "") for x in file_ids]
    p287 = [x for x in file_ids if x.split("_")[0]=='p287']
    reversed_ids = deepcopy(file_ids)
    reversed_ids.reverse()
    matched = {}
    idx = 0
    for i in range(len(file_ids)):
        if file_ids[i].split("_")[0]=='p362':
            matched[file_ids[i]] = p287[idx]
            idx += 1
        else:
            matched[file_ids[i]] = reversed_ids[i]

    return matched

def unseen_files(manifests):
    file_ids = [x['audio_filepath'] for x in manifests]
    file_ids = [x.split("/")[-1].replace(".wav", "") for x in file_ids]

    matched = {}

    unseen = "path-to-unseen-files"
    unseen = [json.loads(x) for x in open(unseen).readlines()]

    random_choose = torch.randint(len(unseen), (1, len(file_ids)))[0]
    assert len(file_ids)==random_choose.size(0)
    for i, idx in enumerate(random_choose):
        idx = idx.data.item()
        x = unseen[idx]
        matched[file_ids[i]] = x['audio_filepath'].split("/")[-1].replace(".wav", "")

    return matched

@hydra_runner(config_path="conf", config_name='infer_transfer')
def main(cfg):
    device = cfg.device

    manifests = [json.loads(x) for x in open(cfg.manifest_path).readlines()]
    # matching files for transfer
    matched_file = match_file_for_transfer(manifests)
    # matched_file = unseen_files(manifests)

    model = VitsModel.load_from_checkpoint(cfg.checkpoint_path).to(device)

    result_dir = Path("result")
    # result_dir = result_dir / cfg.checkpoint_path.split("/")[1] / cfg.checkpoint_path.split("/")[-1]
    result_dir = result_dir / "unseen" / cfg.checkpoint_path.split("/")[1] / cfg.checkpoint_path.split("/")[-1]

    if not result_dir.exists():
        result_dir.mkdir(parents=True, exist_ok=True)

    times = []
    rtfs = []

    for x in tqdm(manifests):
        file_id = x['audio_filepath'].split("/")[-1].replace(".wav", "")
        tokenized = model.tokenizer(x['normalized_text'])
        tokens = torch.tensor(tokenized).long().unsqueeze(0).to(device)
        tokens_length = torch.tensor(len(tokenized)).long().unsqueeze(0).to(device)

        codec, codec_len, spec, spec_len = None, None, None, None
        if 'codec' in cfg.checkpoint_path.lower():
            codec_path = matched_file[file_id] + ".pt"
            codec_path = Path(cfg.sup_data_path) / "encodec" / codec_path
            codec = torch.load(codec_path).long().unsqueeze(0).to(device)
            codec_len = torch.tensor(codec.size(2)).long().unsqueeze(0).to(device)
        else:
            processor = model.audio_to_melspec_processor
            ref_path = f"{matched_file[file_id].split('_')[0]}/{matched_file[file_id]}.wav"
            ref_path = Path(cfg.sup_data_path) / ref_path
            audio, sr = torchaudio.load(ref_path)
            spec, spec_len = processor(audio, torch.tensor([audio.size(1)]),linear_spec=True)
            spec = spec.to(device)
            spec_len = spec_len.to(device)

        spk_id = torch.tensor(x['speaker']).long().unsqueeze(0).to(device)

        start_time = time.perf_counter()
        if codec is not None:
            if isinstance(model.net_g.ref_encoder, GlobalStyleTokenForCodec):
                generator = model.net_g.float()
                wav, _, _, _ = generator.infer(tokens, tokens_length, speakers=spk_id, ref_spec=codec.float(),
                                                 ref_spec_lens=codec_len)
            else:
                wav, _, _, _ = model.net_g.infer(tokens, tokens_length, speakers=spk_id, ref_spec=codec, ref_spec_lens=codec_len)
        elif spec is not None:
            wav, _, _, _ = model.net_g.infer(tokens, tokens_length, speakers=spk_id, ref_spec=spec,
                                             ref_spec_lens=spec_len)
        # wav_len_sec = int(wav.size(-1)) / 24000
        points_per_second = int(wav.size(-1)) / (time.perf_counter() - start_time)
        # rtf = (time.perf_counter() - start_time) / int(wav.size(-1))
        # wav = wav[0][0].detach().cpu().numpy()
        # print(f"inference speed ={per_frame_speed:1.2f} / sec")
        times.append(points_per_second)
        # rtfs.append(rtf)
        # result_filename = file_id + "_" + matched_file[file_id] + ".wav"
        # result_filename = result_dir / result_filename
        # sf.write(result_filename, wav, samplerate=cfg.sample_rate, format="WAV", subtype="PCM_16")
    print(cfg.checkpoint_path.split("/")[1])
    print(f"average inference speed: {mean(times)}")
    # print(f"average RTFs: {mean(rtfs)}")


if __name__ == "__main__":
    main()