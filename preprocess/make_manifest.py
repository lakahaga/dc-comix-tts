import json
import os
import subprocess
from pathlib import Path
import fnmatch
import multiprocessing
import functools
from tqdm import tqdm


def speaker_mapping():
    VCTK = Path("VCTK-Corpus/txt")

    speakers = sorted(os.listdir(VCTK))

    speaker_map = {}
    for i, x in enumerate(speakers):
        speaker_map[x] = i
    # speaker_map = [{'id': x, 'num_id': i} for i, x in enumerate(speakers)]

    if not os.path.exists("data/vctk"):
        os.mkdir("data/vctk")

    json.dump(speaker_map, open("data/vctk/speaker_map.json",'w'), ensure_ascii=False, indent=1)





def __process_transcript(file_path: str):
    entries = []
    with open(file_path, encoding="utf-8") as fin:
        text = fin.readlines()[0].strip()

        wav_file = file_path.replace(".txt", ".wav")
        wav_file = wav_file.replace("/txt/", "/wav22/")
        speaker_id = file_path.split('/')[-2]
        assert os.path.exists(wav_file), f"{wav_file} not found!"
        duration = subprocess.check_output(f"soxi -D {wav_file}", shell=True)
        entry = {
            'audio_filepath': os.path.abspath(wav_file),
            'duration': float(duration),
            'text': text,
            'speaker': speaker_map[speaker_id],
        }

        entries.append(entry)

    return entries


def make_manifest():
    num_workers = 4

    VCTK = Path("VCTK-Corpus")

    txt_dir = VCTK / "txt"
    manifests = []
    files = []

    for root, dirnames, filenames in os.walk(txt_dir):
        # we will use normalized text provided by the original dataset
        for filename in fnmatch.filter(filenames, '*.txt'):
            files.append(os.path.join(root, filename))

    with multiprocessing.Pool(num_workers) as p:
        processing_func = functools.partial(__process_transcript)
        results = p.imap(processing_func, files)
        for result in tqdm(results, total=len(files)):
            manifests.extend(result)

    if not os.path.exists("data/vctk"):
        os.mkdir("data/vctk")
    manifest_file = "data/vctk/all_manifest.json"
    with open(manifest_file, 'w') as fout:
        for m in manifests:
            fout.write(json.dumps(m) + '\n')


def split_manifest():
    all = [json.loads(x) for x in open('data/vctk/all_manifest.json').readlines()]

    speaker_split = {}
    for x in all:
        if x['speaker'] not in speaker_split.keys():
            speaker_split[x['speaker']] = []
        speaker_split[x['speaker']].append(x)
    print(len(speaker_split.keys()))
    total_train, total_dev, total_eval = [], [], []
    for spk, all in tqdm(speaker_split.items()):
        train = int(len(all) * 0.8)
        deveval = len(all) - train
        if deveval % 2 != 0:
            dev = int(deveval / 2) + 1
        else:
            dev = int(deveval / 2)
        eval = deveval - dev
        assert train + dev + eval == len(all)

        train_manifest = all[:train]
        dev_manifest = all[train: train + dev]
        eval_manifest = all[train + dev:]
        assert len(train_manifest) + len(dev_manifest) + len(eval_manifest) == len(all)

        total_train.extend(train_manifest)
        total_dev.extend(dev_manifest)
        total_eval.extend(eval_manifest)

    if not os.path.exists("data/vctk"):
        os.mkdir("data/vctk")
    with open("data/vctk/train_manifest.json", 'w') as f:
        for x in total_train:
            f.write(json.dumps(x) + "\n")
    with open("data/vctk/valid_manifest.json", 'w') as f:
        for x in total_dev:
            f.write(json.dumps(x) + "\n")
    with open("data/vctk/test_manifest.json", 'w') as f:
        for x in total_eval:
            f.write(json.dumps(x) + "\n")
    print(f"Train dataset: {len(total_train)}")
    print(f"Valid dataset: {len(total_dev)}")
    print(f"Test dataset: {len(total_eval)}")
