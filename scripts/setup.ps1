$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPath = Join-Path $ProjectRoot ".venv"
$PythonExe = Join-Path $VenvPath "Scripts\python.exe"

if (-not (Test-Path $PythonExe)) {
    py -3.11 -m venv $VenvPath
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

& $PythonExe -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $PythonExe -m pip install wheel
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $PythonExe -m pip install --no-build-isolation -e "$ProjectRoot[dev]"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
