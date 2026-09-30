$ErrorActionPreference = 'Stop'
if (-not (Test-Path 'vendor/ECGFounder/net1d.py')) {
    git clone https://github.com/PKUDigitalHealth/ECGFounder.git vendor/ECGFounder
    if ($LASTEXITCODE -ne 0) { throw 'Cannot download official source' }
}
git -C vendor/ECGFounder checkout 04edac702b61c91face519774ddcc0cd712fef23
if ($LASTEXITCODE -ne 0) { throw 'Cannot pin official source' }
python -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Cannot install dependencies' }
