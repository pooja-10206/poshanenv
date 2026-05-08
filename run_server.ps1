param(
  [int]$Port = 7860
)

$ErrorActionPreference = "Stop"

Push-Location (Join-Path $PSScriptRoot "server")
try {
  python -m pip install --upgrade pip
  python -m pip install --prefer-binary -r .\requirements.txt
  python -m uvicorn main:app --host 127.0.0.1 --port $Port --reload
} finally {
  Pop-Location
}

