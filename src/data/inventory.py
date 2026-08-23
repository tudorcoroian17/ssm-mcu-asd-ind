import re
import soundfile as sf
import pandas as pd

from pathlib import Path

MACHINE_TYPE = 'ToyCar'
DATASET_ROOT = Path(f'/mnt/d/Tudor/Master/Disertatie/{MACHINE_TYPE}-ToyADMOS-DS')
KEEP_CHANNEL = 'ch1'
SOURCE_FOLDERS = [
    ('AnomalousSound_IND', 'anomaly', 'IND'),
    ('NormalSound_IND', 'normal', 'IND'),
]
CHANNEL_RE = re.compile(r'_ch(\d)_')

rows = []
skipped_other_channel = 0

for case_dir in sorted(DATASET_ROOT.glob('case*')):
    case_id = int(case_dir.name.replace('case', ''))
    for folder_name, label, source in SOURCE_FOLDERS:
        folder = case_dir / folder_name
        if not folder.exists():
            continue

        for wav_path in sorted(folder.glob('*.wav')):
            m = CHANNEL_RE.search(wav_path.name)
            if not m:
                raise ValueError(f'no channel token found in {wav_path.name}')
            channel = f'ch{m.group(1)}'
            if channel != KEEP_CHANNEL:
                skipped_other_channel += 1
                continue

            info = sf.info(str(wav_path))
            rows.append({
                'path': str(wav_path),
                'case_id': case_id,
                'label': label,
                'source': source,
                'channel': channel,
                'sample_rate': info.samplerate,
                'n_samples': info.frames,
                'duration_s': info.frames / info.samplerate,
                'n_channels': info.channels,
                'machine_type': MACHINE_TYPE,
            })

manifest = pd.DataFrame(rows)
print(f'kept {len(manifest)} files, skipped {skipped_other_channel} (other channels than {KEEP_CHANNEL})')
print(manifest.groupby(['case_id', 'label', 'source']).size())
manifest.to_csv('manifest.csv', index=False)