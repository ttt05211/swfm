param(
    [Parameter(Position=0)]
    [string[]]$PythonArgs = @('tools/real_motion/check_local_cuda_env.py')
)

# Reuse the existing CUDA installation without activating Anaconda base or
# changing the caller's PATH. In Codex, run outside its execution sandbox:
# the same healthy Python DLLs return 0xc0000022 inside that sandbox.
$ErrorActionPreference = 'Stop'
$taskRepo = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$taskInterpreter = Join-Path $taskRepo '.venv-cuda-check/Scripts/python.exe'
$taskCondaEnv = 'F:\anaconda3\envs\yoloe'
if (-not (Test-Path -LiteralPath $taskInterpreter -PathType Leaf)) {
    throw 'Project CUDA interpreter missing; see docs/LOCAL_WINDOWS_CUDA_ENV_CN.md. Do not fall back to base.'
}
if (-not (Test-Path -LiteralPath (Join-Path $taskCondaEnv 'python310.dll') -PathType Leaf)) {
    throw 'Existing yoloe runtime missing; do not silently download or use another environment.'
}
if (-not (Get-Content -LiteralPath (Join-Path $taskRepo '.venv-cuda-check/pyvenv.cfg') |
          Where-Object { $_ -ieq ('home = ' + $taskCondaEnv) })) {
    throw 'Project interpreter does not reference the verified yoloe runtime.'
}
if (-not ('SwfmLocalCudaErrorMode' -as [type])) {
    Add-Type -TypeDefinition @'
using System.Runtime.InteropServices;
public static class SwfmLocalCudaErrorMode {
    [DllImport("kernel32.dll")]
    public static extern uint SetErrorMode(uint mode);
}
'@
}
$taskOldMode = [SwfmLocalCudaErrorMode]::SetErrorMode(3)
$taskKeys = @('PATH','CONDA_PREFIX','PYTHONHOME','PYTHONPATH','PYTHONNOUSERSITE',
              'PYTHONDONTWRITEBYTECODE','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS',
              'PYTEST_DISABLE_PLUGIN_AUTOLOAD')
$taskSavedEnvironment = @{}
foreach ($taskKey in $taskKeys) {
    $taskSavedEnvironment[$taskKey] = [System.Environment]::GetEnvironmentVariable($taskKey, 'Process')
}
Push-Location -LiteralPath $taskRepo
try {
    $taskRestPath = ($env:PATH -split ';' | Where-Object { $_ -and $_ -notmatch '(?i)anaconda3|CUDA' }) -join ';'
    $env:PATH = "$taskCondaEnv;$taskCondaEnv\Library\bin;$taskCondaEnv\DLLs;$taskCondaEnv\Scripts;$taskRestPath"
    $env:CONDA_PREFIX = $taskCondaEnv
    Remove-Item Env:PYTHONHOME,Env:PYTHONPATH -ErrorAction SilentlyContinue
    $env:PYTHONNOUSERSITE = '1'
    $env:PYTHONDONTWRITEBYTECODE = '1'
    $env:OMP_NUM_THREADS = '1'
    $env:MKL_NUM_THREADS = '1'
    $env:OPENBLAS_NUM_THREADS = '1'
    $env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
    & $taskInterpreter @PythonArgs
    $taskCode = $LASTEXITCODE
    if ($taskCode -ne 0) {
        throw ("CUDA command failed (exit 0x{0:X8}). If this is 0xC0000022 in Codex, request execution outside the sandbox; do not reinstall CUDA or retry base." -f $taskCode)
    }
}
finally {
    foreach ($taskKey in $taskKeys) {
        [System.Environment]::SetEnvironmentVariable($taskKey, $taskSavedEnvironment[$taskKey], 'Process')
    }
    Pop-Location
    [void][SwfmLocalCudaErrorMode]::SetErrorMode($taskOldMode)
}
