param(
    [ValidateSet("open", "predict")]
    [string]$Command,
    [string]$ClassName
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($ClassName)) {
    throw "Provide a class name like: .\demo.ps1 open dog"
}

$classDir = Join-Path "data\raw" $ClassName
if (-not (Test-Path $classDir)) {
    throw "Class folder not found: $classDir"
}

$image = Get-ChildItem $classDir -File | Select-Object -First 1
if (-not $image) {
    throw "No images found in $classDir"
}

switch ($Command) {
    "open" {
        Invoke-Item $image.FullName
        Write-Host "Opened $($image.FullName)" -ForegroundColor Cyan
    }
    "predict" {
        powershell -ExecutionPolicy Bypass -File .\run.ps1 -Action predict -Image $image.FullName -OpenImage
    }
}
