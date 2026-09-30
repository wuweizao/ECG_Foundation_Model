"""Verify real-data protocol artifacts and official source/weights, no test inference."""
import json
from pathlib import Path
import subprocess
import sys
import ast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import torch

from src.labels import CLASSES
from src.models import build_model
from src.utils import save_json, sha256
from src.dataset import preprocess, LEADS


def main():
    local = json.loads((ROOT/'configs/local.json').read_text())
    expected = 'ee199f3781f4ae1f732973267f003da0a759ea12bddb0dd28a77faa60aca7997'
    assert sha256(local['checkpoint']) == expected
    commit = subprocess.check_output(['git','-C','vendor/ECGFounder','rev-parse','HEAD'],text=True).strip()
    assert commit == '04edac702b61c91face519774ddcc0cd712fef23'
    frames = {name:pd.read_csv(ROOT/f'data/manifests/{name}.csv') for name in ['train','validation','test']}
    sets = {name:set(f.patient_id) for name,f in frames.items()}
    assert not sets['train'] & sets['validation']
    assert not sets['train'] & sets['test']
    assert not sets['validation'] & sets['test']
    for path in (ROOT/'data/manifests').glob('train_f*.csv'):
        sub = pd.read_csv(path)
        assert set(sub.patient_id) <= sets['train']
        expected_frame = frames['train'][frames['train'].patient_id.isin(sub.patient_id)]
        assert set(sub.ecg_id) == set(expected_frame.ecg_id)
        assert (sub[CLASSES].sum() > 0).all()
    torch.manual_seed(42)
    scratch = build_model('scratch')
    torch.manual_seed(42)
    pretrained = build_model('pretrained',local['checkpoint'])
    assert torch.equal(scratch.dense.weight,pretrained.dense.weight)
    assert torch.equal(scratch.dense.bias,pretrained.dense.bias)
    assert list(scratch.state_dict()) == list(pretrained.state_dict())
    assert not torch.equal(scratch.first_conv.conv.weight,pretrained.first_conv.conv.weight)
    probe = build_model('pretrained',local['checkpoint'],True)
    assert all(p.requires_grad == name.startswith('dense.') for name,p in probe.named_parameters())
    # Execute only the upstream filtering function, without unrelated utility imports.
    from scipy.signal import iirnotch, filtfilt, butter, medfilt
    tree = ast.parse((ROOT/'vendor/ECGFounder/util.py').read_text(encoding='utf-8'))
    function = next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='filter_bandpass')
    namespace = dict(np=np,iirnotch=iirnotch,filtfilt=filtfilt,butter=butter,medfilt=medfilt)
    exec(compile(ast.Module(body=[function],type_ignores=[]),'official_filter','exec'),namespace)
    x = np.random.default_rng(42).normal(size=(5000,12))
    official = namespace['filter_bandpass'](x.T,500)
    official = ((official-official.mean())/(official.std()+1e-8)).astype(np.float32)
    np.testing.assert_allclose(preprocess(x,LEADS,500),official,atol=1e-6,rtol=1e-6)
    save_json(ROOT/'results/verification.json',dict(official_checkpoint_verified=True,
                official_source_commit=commit,patient_split_disjoint=True,all_subsets_whole_patient=True,
                classifier_initialization_paired=True,linear_probe_freeze_verified=True,
                preprocessing_matches_official_downstream=True,
                parameters=sum(p.numel() for p in scratch.parameters())))
    print('Artifact and model verification passed')


if __name__ == '__main__':
    main()
