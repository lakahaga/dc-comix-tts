import json
from tqdm import tqdm
from pathlib import Path

from nemo_text_processing.text_normalization.normalize import Normalizer

manifest_path = Path("../data/vctk/24k")
print(manifest_path)
text_normalizer = Normalizer(lang='en', input_case='cased', whitelist="../sup_data/text/whitelist/lj_speech.tsv")
text_norm_kwargs = {"verbose": False, "punct_pre_process": True, "punct_post_process": True}
# for split in ['train', 'valid']:
for split in ['test']:
    manifest_file = manifest_path / f"{split}_manifest.json"
    normalized_manifest = []
    for line in tqdm(open(manifest_file).readlines()):
        item = json.loads(line)

        if 'normalized_text' in item:
            continue
        elif 'text' in item:
            item['normalized_text'] = text_normalizer.normalize(item['text'], **text_norm_kwargs)
        normalized_manifest.append(item)
    normalized_path = manifest_path / "text_normalized"
    if not normalized_path.exists():
        normalized_path.mkdir(exist_ok=True, parents=True)
    normalized_path = manifest_path / "text_normalized" / f"{split}_manifest.json"

    with open(normalized_path, 'w') as fp:
        fp.writelines([json.dumps(x, ensure_ascii=False)+'\n' for x in normalized_manifest])