param(
    [ValidateSet("train", "evaluate", "predict")]
    [string]$Action = "train",
    [string]$Image = "",
    [switch]$OpenImage
)

$ErrorActionPreference = "Stop"

function Invoke-Step {
    param(
        [string]$Title,
        [string]$Command
    )

    Write-Host ""
    Write-Host "== $Title ==" -ForegroundColor Cyan
    Write-Host $Command -ForegroundColor DarkGray
    Invoke-Expression $Command
}

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    throw "Virtual environment not found. Create it first with: python -m venv .venv"
}

switch ($Action) {
    "train" {
        Invoke-Step "Training model" ".venv\Scripts\python -m src.image_classification.train"
    }
    "evaluate" {
        Invoke-Step "Evaluating model" ".venv\Scripts\python -m src.image_classification.evaluate"
    }
    "predict" {
        if ([string]::IsNullOrWhiteSpace($Image)) {
            throw "Provide an image path with -Image for prediction."
        }
        if ($OpenImage) {
            Invoke-Item $Image
        }
        Invoke-Step "Running prediction" ".venv\Scripts\python -m src.image_classification.predict --image `"$Image`""
    }
}
