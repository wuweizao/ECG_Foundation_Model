"""Verify all retained waveform/header files against PhysioNet's release manifest."""
import hashlib
import json
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd
import requests
from src.utils import save_json,sha256


def main():
    local = json.loads((ROOT/'configs/local.json').read_text())
    root = Path(local['data_root'])
    url = 'https://physionet.org/files/ptb-xl/1.0.3/SHA256SUMS.txt'
    response = requests.get(url,timeout=120)
    response.raise_for_status()
    checksums = {}
    for line in response.text.splitlines():
        checksum,name = line.split(maxsplit=1)
        checksums[name.lstrip('*')] = checksum
    files = ['ptbxl_database.csv','scp_statements.csv']
    for split in ['train','validation','test']:
        frame = pd.read_csv(ROOT/f'data/manifests/{split}.csv')
        for name in frame.filename_hr:
            files.extend([name+'.hea',name+'.dat'])
    def verify(name):
        if sha256(root/name) != checksums[name]:
            raise ValueError(f'PhysioNet checksum mismatch: {name}')
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(verify,files))
    save_json(ROOT/'results/data_integrity.json',dict(source=url,release='1.0.3',
              checksum_manifest_sha256=hashlib.sha256(response.content).hexdigest(),
              files_verified=len(files),all_retained_waveforms_verified=True))
    print(f'Verified {len(files)} source files against official PhysioNet SHA256SUMS')


if __name__ == '__main__':
    main()
