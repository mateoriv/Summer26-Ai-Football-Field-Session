# PyInstaller Build Instructions

The repository includes a ready-to-run PyInstaller specification (`hudl_ai.spec`) that turns `app/application.py` into a single-file desktop binary. The spec bundles the runtime Python interpreter, the Qt dependencies, the YOLO model weights, and the cached JSON artifacts so the UI can run without cloning the repo on the target machine. This document explains how to reproduce that build from a clean checkout.

---

## 1. Prerequisites

| Requirement | Notes |
| --- | --- |
| **Python 3.10+** | Build on the same OS/architecture you plan to distribute for—PyInstaller cannot cross-compile. |
| **pip + venv** | Recommended to isolate dependencies. |
| **Git LFS** | Required to download YOLO weight files in `yolo_models/`. Run `git lfs install && git lfs pull` after cloning. |
| **Xcode Command Line Tools / Build Tools** | macOS needs `xcode-select --install`; Windows needs the MSVC build toolchain (via Visual Studio or Build Tools installer). |

---

## 2. Prepare the Environment

1. **Clone and hydrate LFS artifacts**
   ```bash
   git clone https://github.com/<your-org>/Fall25-Ai-Football-Field-Session.git
   cd Fall25-Ai-Football-Field-Session
   git lfs install
   git lfs pull  # fetches yolo_models/*.pt and any other large weights
   ```
2. **(Optional) Create/activate a virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```
3. **Install runtime dependencies plus the build toolchain**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   pip install pyinstaller==6.11.0 pyinstaller-hooks-contrib==2024.11
   ```
   PyInstaller 6.11 ships the fixes needed for PySide6 6.10.0; the hooks contrib package ensures Qt plugins and torch/ultralytics bits are pulled in automatically.

---

## 3. What the Spec File Bundles

`hudl_ai.spec` lives in the repo root and already encodes all the PyInstaller options:

- Entry point: `app/application.py`
- `--onefile` bundle (no `COLLECT` stage) so the output is a single executable.
- Automatic inclusion of PySide6 binaries/translations via `collect_all("PySide6")`.
- Asset directory trees copied verbatim into the bundle (the spec walks each folder with `collect_directory()`), so whatever exists at build time is shipped inside the binary:
  - `app/` – ensures raw `.py` modules are present for any subprocesses or dynamic loading.
  - `yolo_models/` – YOLO weight checkpoints (downloaded through Git LFS).
  - `cache/` – Sample detection/homography JSONs so the UI boots with data.
  - `scripts/` – Helper pipelines invoked from the UI via `subprocess`.
  - `CNN/` – Static processing helper called after homography.

Edit the `asset_dirs` list in the spec if you add or remove directories that must ship with the app.

---

## 4. Run the Build

From the project root:

```bash
pyinstaller --noconfirm --clean hudl_ai.spec
```

- `--clean` removes stale analysis caches that can otherwise reference old modules.
- The build copies everything into `build/` during compilation and writes the final binary to `dist/`.
- Expect the build to take several minutes because PyInstaller needs to bundle PySide6, torch, and the large `.pt` model files.

Resulting artifacts:

| Platform | Output | Notes |
| --- | --- | --- |
| macOS/Linux | `dist/hudl_ai` | Mark executable if needed: `chmod +x dist/hudl_ai`. |
| Windows | `dist/hudl_ai.exe` | Can be launched directly or codesigned before distribution. |

---

## 5. Verifying the Binary

1. **Run from the command line** (ensure the same virtual environment is active so the bundled scripts that call `python` find the interpreter):
   ```bash
   ./dist/hudl_ai            # macOS/Linux
   .\dist\hudl_ai.exe        # Windows (PowerShell or cmd)
   ```
2. **Open the File → Open Folder** menu and point at an example session (e.g., `CSAIQBPSOT/`) to confirm the cached detections load.
3. **Optional:** Trigger a processing step to ensure the subprocess-based scripts (`scripts/*.py` and `CNN/staticProcess.py`) execute. These helpers are included in the bundle, but they are launched via the system Python executable, so the same environment (with `requirements.txt` installed) must remain available on the target machine.

---

## 6. Distribution Tips & Limitations

- **Cache persistence:** PyInstaller’s `--onefile` mode extracts the bundle to a temporary directory on launch. The `cache/` folder that ships with the binary is therefore recreated on every run and removed when the process ends. If you need long-lived caches, point the UI at an external working directory that already contains your session files, or modify the application to read/write caches from a user-writable path outside the bundle.
- **Binary size:** Bundling `torch`, `ultralytics`, and multiple YOLO weights easily pushes the executable beyond 3 GB. Compressing or pruning unused weights before building dramatically shortens build time.
- **Platform-specific builds:** Run PyInstaller on macOS to create `.app`/Mach-O binaries, on Windows for `.exe`, etc. Reuse of the spec file is fine, but the compiled output is not cross-platform.
- **Subprocess helpers still expect Python:** The GUI kicks off detection pipelines by running `python scripts/...`. Those subprocesses do **not** reuse the interpreter embedded inside the PyInstaller EXE. Ship the same Python environment alongside the binary (or refactor the app to call directly into the packaged modules) if you need these features on a machine without Python installed.
- **Qt plugin issues:** If you see errors like “This application failed to start because no Qt platform plugin could be initialized”, delete `build/` + `dist/`, reinstall `pyinstaller-hooks-contrib`, and rebuild so that the PySide6 hooks run cleanly.
- **`warn-hudl_ai.txt` mentions TensorFlow (or other modules):** PyInstaller lists optional imports pulled in by PyTorch/Ultralytics even if they are never used. Unless you rely on those extras at runtime, the warnings are benign.

---

## 7. Troubleshooting Checklist

- **Missing models in the bundle:** Re-run `git lfs pull` and rebuild. The spec only copies `yolo_models/` if the directory exists.
- **Unsigned macOS binary warnings:** Use `codesign --deep --force --options runtime --sign "<identity>" dist/hudl_ai` and then `spctl --add dist/hudl_ai`.
- **Runtime crashes during import:** Add `--log-level=DEBUG` to the PyInstaller command to surface hidden imports, then append them to `hiddenimports` in `hudl_ai.spec`.
- **Faster iteration:** During development you can temporarily switch to `pyinstaller hudl_ai.spec --onedir` by adding a `COLLECT` step to the spec—this skips the onefile compression and speeds up rebuilds.

With the steps above you can reliably regenerate the single-file binary with all YOLO model weights and cached artifacts baked in. Update this document whenever the asset list or build flags change so the next build is just a single command away.
