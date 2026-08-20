$ErrorActionPreference = "Stop"
$MinimumMemoryGB = 10
$InstallRoot = if ($env:ORBIT_INSTALL_ROOT) { $env:ORBIT_INSTALL_ROOT } else { Join-Path $env:LOCALAPPDATA "Orbit" }
$RuntimeDir = Join-Path $InstallRoot "runtime"
$ArchiveUrl = if ($env:ORBIT_ARCHIVE_URL) { $env:ORBIT_ARCHIVE_URL } else { "https://github.com/ljcccc999/orbit/archive/refs/heads/main.zip" }
$TempDir = Join-Path ([IO.Path]::GetTempPath()) ("orbit-install-" + [guid]::NewGuid())

try {
    $memory = (Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB
    if ($memory -le $MinimumMemoryGB) { throw "Orbit requires more than 10 GB of memory. This computer reports $([math]::Round($memory, 1)) GB." }
    New-Item -ItemType Directory -Force -Path $InstallRoot, $TempDir | Out-Null

    $sourceDir = $env:ORBIT_SOURCE_DIR
    if (-not $sourceDir -or -not (Test-Path (Join-Path $sourceDir "pyproject.toml"))) {
        Write-Host "Downloading Orbit..."
        $archive = Join-Path $TempDir "orbit.zip"
        for ($attempt = 1; $attempt -le 5; $attempt++) {
            try { Invoke-WebRequest -Uri $ArchiveUrl -OutFile $archive -TimeoutSec 300; break }
            catch { if ($attempt -eq 5) { throw }; Start-Sleep -Seconds ($attempt * 2) }
        }
        Expand-Archive -Path $archive -DestinationPath $TempDir
        $project = Get-ChildItem $TempDir -Filter pyproject.toml -Recurse | Select-Object -First 1
        if (-not $project) { throw "The Orbit archive is invalid." }
        $sourceDir = $project.Directory.FullName
    }

    $pythonOkay = Test-Path (Join-Path $RuntimeDir "Scripts\python.exe")
    if ($pythonOkay) {
        & (Join-Path $RuntimeDir "Scripts\python.exe") -c "import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)"
        $pythonOkay = $LASTEXITCODE -eq 0
    }
    if (-not $pythonOkay) {
        Write-Host "Preparing a private Python 3.11 runtime..."
        $tools = Join-Path $InstallRoot "tools"
        $env:UV_UNMANAGED_INSTALL = $tools
        Invoke-Expression (Invoke-RestMethod "https://astral.sh/uv/install.ps1")
        $newRuntime = Join-Path $InstallRoot ("runtime.new." + $PID)
        & (Join-Path $tools "uv.exe") venv --python 3.11 $newRuntime
        & (Join-Path $newRuntime "Scripts\python.exe") -m pip install --retries 8 --timeout 60 --upgrade pip $sourceDir
        if (Test-Path $RuntimeDir) { Remove-Item -Recurse -Force $RuntimeDir }
        Move-Item $newRuntime $RuntimeDir
    } else {
        & (Join-Path $RuntimeDir "Scripts\python.exe") -m pip install --retries 8 --timeout 60 --upgrade pip $sourceDir
    }
    & (Join-Path $RuntimeDir "Scripts\orbit.exe") start
    Write-Host "Orbit is ready at http://127.0.0.1:8765"
}
finally {
    if (Test-Path $TempDir) { Remove-Item -Recurse -Force $TempDir }
}
