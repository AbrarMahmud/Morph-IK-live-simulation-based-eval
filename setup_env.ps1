# Windows PowerShell: creates the dedicated virtual environment (.venv) and installs the demo.
#   .\setup_env.ps1           # default PyPI torch
#   .\setup_env.ps1 -Cpu      # CPU-only torch wheel
param([switch]$Cpu)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$py = "py -3.11"
try { & py -3.11 --version | Out-Null } catch { $py = "python" }
Write-Host "[setup] creating .venv with $py"
Invoke-Expression "$py -m venv .venv"
& .\.venv\Scripts\python.exe -m pip install --upgrade pip wheel
if ($Cpu) { & .\.venv\Scripts\pip.exe install torch --index-url https://download.pytorch.org/whl/cpu }
& .\.venv\Scripts\pip.exe install -r requirements.txt
& .\.venv\Scripts\python.exe -m graphik_demo selftest
Write-Host "`n[setup] done. Activate with:  .\.venv\Scripts\Activate.ps1"
