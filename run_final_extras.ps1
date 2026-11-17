# Extra experiments requested at the final review (PowerShell version of run_final_extras.sh).
#   1. patch placement sweep (evaluation only, ~15 min)
#   2. total-variation ablation: retrain the patch with lambda_TV = 0, 1, 5, 10 (~1.1 h each),
#      then test each patch at low resolution and strong JPEG.
# Safe to re-run: finished steps are skipped.
#   powershell -ExecutionPolicy Bypass -File run_final_extras.ps1
Set-Location $PSScriptRoot
$py = 'C:\Python314\python.exe'
$logs = 'runs\extra_logs'
New-Item -ItemType Directory -Force $logs | Out-Null
$env:PYTHONUNBUFFERED = '1'

function Run($log, [string[]]$argv) {
    $p = Start-Process -FilePath $py -ArgumentList $argv -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput "$logs\$log.out.log" -RedirectStandardError "$logs\$log.err.log"
    "$(Get-Date -Format HH:mm) $log exit $($p.ExitCode)" | Add-Content "$logs\progress.txt"
}

$pm = 'runs\placement\yolov8s\placement_metrics.json'
if (-not ((Test-Path $pm) -and (Select-String -Path $pm -Pattern '"peak"' -Quiet))) {
    Run 'placement' @('src\placement_study.py')
}

foreach ($tv in @('0', '1', '5', '10')) {
    if (-not (Test-Path "runs\attack\tv_$tv\attack_metrics.json")) {
        $a = @('src\attack_patch.py', '--tv-weight', $tv, '--name', "tv_$tv")
        if (Test-Path "runs\attack\tv_$tv\train_log.json") { $a += '--resume' }
        Run "tv_$tv" $a
    }
    if (-not (Test-Path "runs\robustness\tv_$tv\robustness_metrics.json")) {
        Run "rob_tv_$tv" @('src\robustness.py', '--patch', "runs/attack/tv_$tv/patch.png", '--name', "tv_$tv",
                           '--factors', 'resolution', 'jpeg_quality')
    }
}
"$(Get-Date -Format HH:mm) ALL_DONE" | Add-Content "$logs\progress.txt"
