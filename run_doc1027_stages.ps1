param(
    [Parameter(Mandatory = $true)][string]$Manifest,
    [Parameter(Mandatory = $true)][string]$Support,
    [Parameter(Mandatory = $true)][string]$BaselineProvenance,
    [string]$OutputRoot = "runs/doc1027_staged",
    [string]$Python = "python",
    [string]$Device = "cuda",
    [ValidateSet("dense_only", "ratio_control", "feature_synthetic", "image_synthetic", "joint_synthetic", "joint_rank")]
    [string[]]$Stages = @("dense_only", "ratio_control", "feature_synthetic", "image_synthetic", "joint_synthetic", "joint_rank"),
    [switch]$Resume,
    [switch]$PrintOnly
)

$ErrorActionPreference = "Stop"
$manifestPath = (Resolve-Path -LiteralPath $Manifest).Path
$supportPath = (Resolve-Path -LiteralPath $Support).Path
$baselinePath = (Resolve-Path -LiteralPath $BaselineProvenance).Path
$outputPath = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutputRoot)

Push-Location -LiteralPath $PSScriptRoot
try {
    foreach ($stage in $Stages) {
        $configPath = Join-Path $PSScriptRoot "configs/doc1027_staged/$stage.json"
        $stageOutput = Join-Path $outputPath $stage
        $arguments = @(
            "run_brain_novelty.py", "train",
            "--manifest", $manifestPath,
            "--support", $supportPath,
            "--baseline-provenance", $baselinePath,
            "--config", $configPath,
            "--output", $stageOutput,
            "--device", $Device
        )
        if ($Resume) { $arguments += "--resume" }
        Write-Host "[stage] $stage"
        if ($PrintOnly) {
            Write-Host ($Python + " " + (ConvertTo-Json -InputObject $arguments -Compress))
            continue
        }
        & $Python @arguments
        if ($LASTEXITCODE -ne 0) {
            throw "Stage '$stage' failed (exit $LASTEXITCODE). Later stages were not started."
        }
    }
}
finally {
    Pop-Location
}
