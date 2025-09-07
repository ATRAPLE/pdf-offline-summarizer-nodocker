$here = Split-Path -Parent $MyInvocation.MyCommand.Path
cd $here
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
if (-not (Test-Path ".env")) { Copy-Item ".env.sample" ".env" }
uvicorn app.main:app --host 0.0.0.0 --port 8080
