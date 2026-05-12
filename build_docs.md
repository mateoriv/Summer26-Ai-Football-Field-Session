# Build Instructions

The repository builds a single-file Windows executable (`hudl_ai.exe`) using **PyInstaller** and the spec file at the repo root (`hudl_ai.spec`).

---

## 1. Prerequisites

| Requirement | Notes |
| --- | --- |
| **Python 3.10+** | Build on the same OS/architecture you plan to distribute for — PyInstaller cannot cross-compile. |
| **pip + venv** | Recommended to isolate dependencies. |
| **Git LFS** | Required to download YOLO weight files in `yolo_models/`. Run `git lfs install && git lfs pull` after cloning. |
| **MSVC Build Tools (Windows)** | Required for some native wheels; install via [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022). |

---

## 2. Prepare the Environment

```bash
git clone https://github.com/<your-org>/Fall25-Ai-Football-Field-Session.git
cd Fall25-Ai-Football-Field-Session
git lfs install
git lfs pull            # fetches yolo_models/*.pt and other large weights

python -m venv .venv
.\.venv\Scripts\activate   # PowerShell on Windows
# source .venv/bin/activate  # macOS / Linux

pip install --upgrade pip
pip install -r requirements.txt
pip install pyinstaller pyinstaller-hooks-contrib
```

> Tip: installing **CPU-only torch** in the build env keeps the EXE much smaller and avoids CUDA DLL dependency chains at runtime.
> ```bash
> pip uninstall -y torch torchvision
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
> ```

---

## 3. What the Spec File Bundles

`hudl_ai.spec` lives in the repo root and is intentionally minimal:

- **Entry point:** `app/application.py`
- **Single-file EXE:** `EXE(...)` is called with `a.binaries, a.zipfiles, a.datas` — no separate `COLLECT`/onedir stage.
- **Bundled asset directories** (`asset_dirs` list in the spec):
  - `app/` — UI modules and runtime helpers
  - `scripts/` — pipeline scripts (Player Detection through Static Process)
  - `models/` — offense-position model + metadata used by Step 7
  - `yolo_models/` — YOLO weight checkpoints (downloaded through Git LFS)
- **Hidden imports:** `darkdetect`, `cv2`, `numpy`, `ultralytics`, `ultralytics.models`, `ultralytics.utils`, `torch`, `scipy`, `scipy.spatial`, `scipy.spatial.distance`, `pdb`.
- **Excludes** (minimal, by design): `tkinter`, `modelTraining`.

> The spec deliberately keeps `excludes` short. Aggressive excludes of torch submodules (e.g. `torch.distributed`, `torch.testing`) or stdlib modules (e.g. `pdb`) caused runtime `ModuleNotFoundError` failures because PyTorch imports them transitively at startup.

Edit `asset_dirs` in the spec if you add or remove directories that must ship with the app.

---

## 4. Run the Build

From the project root:

```bash
pyinstaller --noconfirm --clean hudl_ai.spec
```

- `--clean` removes stale analysis caches that can otherwise reference old modules.
- The build writes intermediate artifacts to `build/` and the final binary to `dist/hudl_ai.exe`.
- Expect several minutes because PyInstaller has to bundle PySide6, torch, and the YOLO weights.

| Platform | Output | Notes |
| --- | --- | --- |
| Windows | `dist/hudl_ai.exe` | Single file. Copy anywhere. |
| macOS/Linux | `dist/hudl_ai` | Mark executable if needed: `chmod +x dist/hudl_ai`. |

---

## 5. Verifying the Binary

1. **Run from the command line** (enables stdout/stderr visibility if `console=True` in the spec):
   ```bash
   .\dist\hudl_ai.exe
   ```
2. **Open the File → Open Folder** menu and point at a session folder (e.g., `FootballFootage/DemoFolder/`) to confirm video + CSV load correctly.
3. **Run a processing step** to verify subprocess script execution. When frozen, the GUI re-invokes the EXE with the env var `PYINSTALLER_RUN_SCRIPT` pointing at a `scripts/*.py` file, and the frozen interpreter runs that script. No system Python install is required on the target machine.

---

## 6. Distribution Tips & Limitations

- **Cache location:** When frozen, the app writes runtime cache (`cache/`) next to the EXE via `fileAccess.get_cache_dir()`. It is **not** bundled inside the EXE.
- **Debug console:** Set `console=True` in the `EXE(...)` block of `hudl_ai.spec` to see logs when launching the EXE.
- **Binary size:** Full CUDA torch can push the bundle past PyInstaller's 4 GB single-file limit (`struct.error: 'I' format requires 0 <= number <= 4294967295`). Either install CPU-only torch in the build env (recommended) or switch to a onedir build by adding a `COLLECT` stage to the spec.
- **Platform-specific builds:** Build on each target OS — PyInstaller cannot cross-compile.
- **Subprocess scripts reuse the frozen EXE:** The GUI launches scripts by re-invoking `sys.executable` with `PYINSTALLER_RUN_SCRIPT` (see `build_script_command` in `app/processingDialog.py`). Every dependency used by the scripts (including `scipy`) must therefore stay bundled — never add script-only dependencies to `excludes`.
- **`warn-hudl_ai.txt` mentions TensorFlow / extras:** PyInstaller lists optional imports pulled in by PyTorch/Ultralytics even if they are never used at runtime. These warnings are benign.

---

## 7. Troubleshooting Checklist

- **`ModuleNotFoundError: No module named 'torch.distributed'` / `'torch.testing'` / `'pdb'`** at runtime: something is being excluded that PyTorch needs. Remove the offending entry from `excludes` and rebuild with `--clean`.
- **`OSError: [WinError 126] ... torch.dll`**: a torch dependency DLL is missing from the bundle. Don't manually filter DLLs in the spec — rebuild against a fresh CPU-only torch wheel.
- **`struct.error: 'I' format requires 0 <= number <= 4294967295`**: the onefile archive exceeded 4 GB. Switch to CPU-only torch or convert the spec to onedir mode (add a `COLLECT` block and use `EXE(pyz, a.scripts, exclude_binaries=True, ...)` plus `COLLECT(exe, a.binaries, a.zipfiles, a.datas, ...)`).
- **Stale build:** if rebuilding doesn't seem to pick up spec changes, delete `build/` and `dist/` and re-run with `--clean`.
- **Missing models in the bundle:** re-run `git lfs pull` and rebuild — the spec only copies `yolo_models/` and `models/` if they exist on disk.
- **Unsigned macOS binary warnings:** `codesign --deep --force --options runtime --sign "<identity>" dist/hudl_ai && spctl --add dist/hudl_ai`.
- **Runtime crashes during import:** add `--log-level=DEBUG` to the PyInstaller command to surface hidden imports, then append them to `hiddenimports` in `hudl_ai.spec`.
