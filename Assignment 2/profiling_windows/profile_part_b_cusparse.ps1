# Part B cuSPARSE profiling: compile, plain run, ncu smoke, ncu export.
# Run from any location; script uses Assignment 2 as project root (parent of profiling_windows).

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot
Write-Host "[1/6] Project root: $ProjectRoot"

Write-Host "[2/6] Checking nvcc and ncu"
& nvcc --version
& ncu --version

Write-Host "[3/6] Compiling Part B cuSPARSE profiling binary"
& nvcc -O3 -lineinfo -std=c++17 -DPART_B_PROFILE_MODE=1 -o part_b_wave_profile_cusparse.exe part_b_wave.cu -lcusparse -lcublas
Get-Item part_b_wave_profile_cusparse.exe

Write-Host "[4/6] Plain execution sanity check"
& .\part_b_wave_profile_cusparse.exe | Tee-Object -FilePath part_b_cusparse_profile_plain.log

Write-Host "[5/6] Nsight Compute smoke (LaunchStats, 1 launch)"
& ncu --section LaunchStats --target-processes application-only --launch-count 1 .\part_b_wave_profile_cusparse.exe 2>&1 | Tee-Object -FilePath part_b_cusparse_profile_ncu_smoke.log

Write-Host "[6/6] Exporting final Part B cuSPARSE report (.ncu-rep)"
& ncu --set speedOfLight --section LaunchStats --target-processes application-only --launch-skip 20 --launch-count 10 --force-overwrite true --export part_b_cusparse_profile .\part_b_wave_profile_cusparse.exe 2>&1 | Tee-Object -FilePath part_b_cusparse_profile_ncu_final.log

Write-Host "Done. Report: part_b_cusparse_profile.ncu-rep"
