"""Build and execute the reader-facing notebook with the current Python kernel."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
runtime = ROOT / '.notebook_runtime'
if runtime.exists():
    sys.path.insert(0,str(runtime))
    os.environ['PYTHONPATH'] = str(runtime) + os.pathsep + str(ROOT)
os.environ['JUPYTER_RUNTIME_DIR'] = str(ROOT / '.jupyter_runtime')
os.environ['IPYTHONDIR'] = str(ROOT / '.ipython')
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpec


def main():
    notebook = nbformat.v4.new_notebook()
    notebook.metadata.kernelspec = dict(display_name='ECG Foundation (Python 3)',language='python',name='python3')
    markdown,code = nbformat.v4.new_markdown_cell,nbformat.v4.new_code_cell
    notebook.cells = [
        markdown('# ECG Foundation Model benchmark\n\n## Goal\nReview reproducible, patient-independent multi-label ECG results. No training or test-set retuning is performed in this notebook.'),
        markdown('## Setup\nRun the completed benchmark first. Sources: PTB-XL 1.0.3, official ECGFounder. Paths are relative to the repository, and all displayed metrics come from saved real experiment outputs.'),
        code("from pathlib import Path\nimport json\nimport pandas as pd\nfrom IPython.display import display, Image\nROOT = Path.cwd()\nif not (ROOT / 'results').exists():\n    ROOT = ROOT.parent\nassert (ROOT / 'results/metrics.csv').exists(), 'Run the complete benchmark first'"),
        markdown('## Steps\n### 1. Inspect the cohort\nOfficial folds 1–8 / 9 / 10; exclude records without mapped diagnostic labels. Patient counts refer to retained eligible ECGs.'),
        code("audit = json.loads((ROOT / 'results/data_audit.json').read_text())\ndisplay(pd.DataFrame(audit['splits']).T[['records','patients']])\nprint('Excluded without diagnostic labels:', audit['excluded_no_diagnostic_label'])"),
        markdown('### 2. Compare actual held-out results\nMean ± sample SD is reported for three seeds at 1–25%. At 100%, SD is undefined because only one seed was run.'),
        code("summary = pd.read_csv(ROOT / 'results/summary.csv')\ndisplay(summary.round(4))"),
        markdown('### 3. Inspect label efficiency\nThe x-axis is the percentage of training patients, not ECG records. Error bars show training/subsampling seed variability.'),
        code("display(Image(filename=str(ROOT / 'figures/label_efficiency.png')))"),
        code("display(Image(filename=str(ROOT / 'figures/auprc.png')))"),
        markdown('### 4. Inspect disease-specific results\nFive independent labels; one ECG can contribute to multiple positive groups.'),
        code("display(Image(filename=str(ROOT / 'figures/per_class.png')))"),
        markdown('## Checks\nAll 26 full-fine-tuning conditions and 7 linear probes must be present. A separate test lock records weights and thresholds before evaluation.'),
        code("metrics = pd.read_csv(ROOT / 'results/metrics.csv')\nassert len(metrics) == 33\nassert not metrics[['model','fraction','seed']].duplicated().any()\nassert metrics['macro_auroc'].between(0,1).all()\nlock = json.loads((ROOT / 'results/test_lock.json').read_text())\nassert sum(r['group']=='main' for r in lock['runs']) == 33\nassert sum(r['group']=='convergence' for r in lock['runs']) == 2\nprint('33 main/probe runs and 2 budget-sensitivity runs locked.')"),
        markdown('## Next Steps\nInterpret label efficiency as a discrete descriptive comparison, not statistical equivalence. Check training-budget diagnostics before claiming convergence. Robustness is post-preprocessing input corruption; patient bootstrap uncertainty is conditional on fitted models. See `experiments/PROTOCOL.md` for limitations.'),
        code("display(json.loads((ROOT / 'results/label_efficiency_gain.json').read_text()))"),
    ]
    nbformat.validate(notebook)
    target = ROOT / 'notebooks/demo.ipynb'
    target.parent.mkdir(exist_ok=True)
    nbformat.write(notebook,target)
    # Use an explicit local kernel; do not install a global kernelspec.
    km = KernelManager(kernel_name='python3')
    km._kernel_spec = KernelSpec(argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
                                display_name='ECG Foundation',language='python')
    client = NotebookClient(notebook,timeout=120,resources={'metadata':{'path':str(ROOT)}},km=km)
    try:
        client.execute()
    finally:
        if km.has_kernel:
            km.shutdown_kernel(now=True)
    nbformat.validate(notebook)
    nbformat.write(notebook,target)
    print('Notebook executed top-to-bottom:',target)


if __name__ == '__main__':
    main()
