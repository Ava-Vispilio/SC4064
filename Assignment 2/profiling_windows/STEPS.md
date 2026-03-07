# Steps to run profiling on Windows

Use these steps on a Windows machine with an NVIDIA GPU (e.g. RTX 2060 Super). You will need to install the CUDA Toolkit and Nsight Compute yourself (download from NVIDIA).

---

## 1. Install CUDA Toolkit

- Download the **CUDA Toolkit** for Windows from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads).
- Run the installer and choose a path (e.g. `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x`).
- Ensure the **bin** directory of the toolkit is on your PATH so that `nvcc` is available in a terminal (the installer may add it; if not, add it manually).

---

## 2. Install Nsight Compute

- Download **Nsight Compute** from [NVIDIA Nsight Compute](https://developer.nvidia.com/nsight-compute).
- Install it (e.g. under `C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.x` or similar).
- Locate the **ncu** executable (often in a subfolder like `2024.x\` inside the install directory). You will need this directory on your PATH so that `ncu` runs from a terminal.

---

## 3. Add tools to PATH (if not already)

- Add the CUDA Toolkit **bin** folder to your system or user PATH (e.g. `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin`).
- Add the Nsight Compute folder that contains `ncu.exe` to your PATH.
- Open a **new** PowerShell or Command Prompt and verify:
  - `nvcc --version`
  - `ncu --version`

---

## 4. Open a terminal in the Assignment 2 folder

- In File Explorer, go to your repo and open the **Assignment 2** folder (the one that contains `part_a_wave.cu`, `part_b_wave.cu`, and the `profiling_windows` subfolder).
- Open PowerShell in that folder (e.g. Shift+Right-click → “Open PowerShell window here”, or run `cd "C:\path\to\SC4064\Assignment 2"` in an already-open PowerShell).

---

## 5. Allow script execution (PowerShell, first time only)

If you get a policy error when running the `.ps1` scripts, run once in that PowerShell (as Administrator if required):

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## 6. Run the profiling scripts

From the **Assignment 2** directory, run one script at a time:

- **Part A (representative kernel):**
  ```powershell
  .\profiling_windows\profile_part_a.ps1
  ```

- **Part B cuSPARSE:**
  ```powershell
  .\profiling_windows\profile_part_b_cusparse.ps1
  ```

- **Part B cuBLAS:**
  ```powershell
  .\profiling_windows\profile_part_b_cublas.ps1
  ```

Each script will: compile the profiling binary, run a plain sanity run, run a short Nsight Compute smoke test, then run the full Nsight Compute export.

---

## 7. Where to find the results

- **Assignment 2** (project root) will contain:
  - `part_a_profile.ncu-rep`, `part_b_cusparse_profile.ncu-rep`, `part_b_cublas_profile.ncu-rep` (after each script finishes).
  - Plain run logs: `part_a_profile_plain.log`, `part_b_cusparse_profile_plain.log`, `part_b_cublas_profile_plain.log`.
  - Nsight logs: `*_ncu_smoke.log`, `*_ncu_final.log`.

Open the `.ncu-rep` files in the Nsight Compute GUI (File → Open) for detailed kernel analysis.

---

## Optional: Run from the profiling_windows folder

If you prefer to run from inside `profiling_windows`, use the parent directory when calling the script so the script still finds the sources:

```powershell
cd "C:\path\to\SC4064\Assignment 2\profiling_windows"
.\profile_part_a.ps1
```

The scripts are written to change to the Assignment 2 directory automatically, so this works the same.
