# Linux VM deployment and PyInstaller release

This repository can prepare a Linux **onedir** release, but this Windows
workspace cannot produce or validate the Ubuntu executable. Build on Linux
matching the VM's Ubuntu release and architecture. Do not use a Windows
PyInstaller executable on Ubuntu.

## Audit and deployment boundary

The active entry point is `app.py`, which starts the existing
`alcon_stream_new.py` pipeline. It initializes Flask/Flask-SocketIO in
threading mode, checks PostgreSQL, creates the AI models, initializes Firebase,
and starts configured camera workers. The refactor packages under `ai/`,
`camera/`, and related directories mostly re-export or isolate code from the
monolith; they are still included in PyInstaller analysis.

## Select source or compiled mode

On Linux, `run.sh` selects a mode explicitly and never falls back to another:

```bash
# Original Python source; requires the project's populated .venv
source .venv/bin/activate
./run.sh source

# Existing PyInstaller onedir executable
./run.sh build

# Rebuild from wave-vms.spec using the project's .venv
./run.sh compile

# Show which modes are available
./run.sh status
```

Source and build modes pass remaining arguments to the application, for
example `./run.sh source --help`. The compile mode passes remaining arguments
to PyInstaller. Source mode invokes `.venv/bin/python app.py`, equivalent to
activating that environment and running `python app.py`.

Install build requirements into the project environment before compiling:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-build.txt
./run.sh compile
```

`run.sh compile` requires Linux and builds the onedir output at
`dist/wave-vms/` with `wave-vms.spec`. Use `build_linux.sh` when you need
explicit target Ubuntu, architecture, and glibc checks plus build metadata.
Neither command deletes the development source.

Both modes default `WAVE_RUNTIME_DIR` to the project root, unless the
environment already sets it. This shares `.env`, camera configuration,
models, face registrations, alert images, and PostgreSQL settings. To share an
external persistent directory instead, set the same absolute path before
either command:

```bash
export WAVE_RUNTIME_DIR=/var/lib/wave-vms
./run.sh source
# Stop source before explicitly choosing the compiled mode:
./run.sh build
```

Direct invocation of `dist/wave-vms/wave-vms` from a development checkout
detects the project root under `dist/wave-vms`; a deployed release should set
`WAVE_RUNTIME_DIR` to the external runtime root. `app.py` acquires a
non-blocking `$WAVE_RUNTIME_DIR/.wave-vms.lock`; another source or compiled
process pointed at the same runtime fails with an explicit error, rather than
starting duplicate camera workers or database writers. The lock file does not
alter the database or other persistent data. The run launcher never stops
another process automatically.

`./run.sh status` reports availability of the source interpreter and compiled
executable. A requested unavailable mode exits with an error instead of
silently selecting the other mode.

| Area | Findings |
| --- | --- |
| Web/API | Flask 3, Flask-CORS, Flask-SocketIO, Socket.IO; default bind is `0.0.0.0:5000`, now configurable with `HOST` and `PORT`. |
| Persistence | Active database helper uses PostgreSQL (`psycopg`), not SQLite. The release installer does not initialize, migrate, or modify PostgreSQL. |
| Models present in repository | `yolo11n.pt`, `yolo26l.pt`, `yolo11m-pose.pt`, `yolo26n_ppe.pt`, `salahyolo26.pt`, `barrel.pt`. All six local weight files load successfully with Ultralytics in the current Windows Python environment. |
| InsightFace models | `buffalo_l` is not in the repository. Provision its model cache separately and verify it on staging. |
| Face security | The active code loads InsightFace `buffalo_l`; this audit did not find a separate anti-spoofing model or anti-spoofing pipeline. |
| Camera and persistent data | Camera definitions, registered face images, alert images, environment, Firebase credentials, and InsightFace model cache are read from the external `WAVE_RUNTIME_DIR`. A sanitized camera template has all cameras disabled and contains no site IPs or camera names. |
| Frontend/Nginx | No React/Vite project, `dist/`, or Nginx configuration is present in this repository. Preserve and inspect the existing VM's frontend and Nginx configuration independently. |
| GPU/native runtime | The requirements select `onnxruntime-gpu==1.19.2`; PyTorch/torchvision are ranged. Driver, CUDA/cuDNN, target glibc, ONNX providers, and actual provider use cannot be confirmed from this Windows workspace. |

PyInstaller is packaging, not encryption. Python bytecode and strings in the
release may be recoverable. Keep release access restricted, run under a
least-privilege account, and keep credentials external.

## Dependencies and native compatibility

Application dependencies are defined in `requirements.txt`. The inspected
workspace uses Python 3.11.9. `requirements-build.txt` pins PyInstaller; the
Linux build also records the resolved package versions in
`DEPENDENCIES.lock`. The project's existing requirements contain version
ranges, so retain and reuse that generated lock (or provide it as
`PIP_CONSTRAINT_FILE`) for repeat builds. Do not accept dependency upgrades
without staging regression tests.

Build and test with the same Ubuntu `VERSION_ID` and CPU architecture as the
target. The release includes Python extension modules, but not NVIDIA drivers.
Check unresolved shared libraries with `ldd`; the Ubuntu VM may also require
system OpenCV/video libraries such as `libgl1`, `libglib2.0-0`, `libgomp1`,
`libstdc++6`, and FFmpeg support. Install only the packages shown to be
required by the target's `ldd`, OpenCV capture tests, and deployment setup.
Do not bundle system NVIDIA drivers.

Before production, record on the VM/build host:

```bash
uname -m
lsb_release -ds
getconf GNU_LIBC_VERSION
python3 --version
nvidia-smi
```

Then on a matching Linux Python environment, verify:

```bash
python -c 'import onnxruntime as o; print(o.get_available_providers())'
```

The app's startup output reports the providers selected for InsightFace. This
must be checked alongside the model's actual session providers; a successful
build alone does not prove GPU inference.

## Build on Linux

On Windows, install WSL2 and an Ubuntu distribution matching the target VM
(replace `22.04` with the target's release):

```powershell
wsl --install -d Ubuntu-22.04
wsl -d Ubuntu-22.04
```

In that Ubuntu shell, install Python 3.11 and build dependencies. For Ubuntu
22.04, Python 3.11 can be installed from the deadsnakes PPA:

```bash
sudo apt-get update
sudo apt-get install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv python3.11-dev \
  build-essential pkg-config libgl1 libglib2.0-0 libgomp1 ffmpeg
```

Then clone/copy the repository into the WSL filesystem (not `/mnt/c` for
performance). To copy the current worktree while omitting local secrets and
caches, replace the source path with the Windows workspace path:

```bash
sudo apt-get install -y rsync
mkdir -p ~/src/wave-vms
rsync -a \
  --exclude='.git/' --exclude='.venv/' --exclude='venv/' \
  --exclude='__pycache__/' --exclude='.pytest_cache/' \
  --exclude='known_faces/' --exclude='alert_images/' \
  --exclude='.env' --exclude='pythonai.json' \
  /mnt/c/path/to/workspace/ ~/src/wave-vms/
cd ~/src/wave-vms
```

Alternatively, copy only tracked files from a clean checkout, then transfer
the intended source changes. Enter the repository root and run the build. The
script intentionally refuses to guess the target release:

```bash
TARGET_UBUNTU_RELEASE=22.04 \
TARGET_ARCH=x86_64 \
TARGET_GLIBC_VERSION="glibc 2.35" \
bash build_linux.sh
```

Replace all three values with the VM's actual `VERSION_ID`, `uname -m`, and
`getconf GNU_LIBC_VERSION` output. The build runs PyInstaller
in onedir mode with UPX disabled. It does not include secrets, camera settings,
face data, event images, a database, or model weights in the executable.
`dist/wave-vms/` is the compiled output. Then generate a versioned release:

```bash
bash scripts/make_release.sh 2026.10.22
```

For a later repeat build in the same Linux environment, constrain dependency
resolution using the recorded lock:

```bash
PIP_CONSTRAINT_FILE=build/linux-dependencies.lock \
  TARGET_UBUNTU_RELEASE=22.04 \
  TARGET_ARCH=x86_64 \
  TARGET_GLIBC_VERSION="glibc 2.35" \
  bash build_linux.sh
```

This creates `release/wave-vms-2026.10.22/`, a tarball, and a detached
`.sha256` file. The release directory contains only the executable/runtime,
approved model weights found in the repository, the sanitized camera template,
deployment scripts, and build/dependency metadata.

The release builder copies the six model weights present in this checkout to
the release's `models/` directory and lists them in `models/ASSET_MANIFEST.txt`.
The installer copies these weights into the external runtime `models/`
directory only when a same-named file is not already there. It never replaces
an existing model. The Linux build and Ubuntu runtime still need model-loading
verification, and the InsightFace `buffalo_l` cache must be provisioned
separately. Do not substitute unrelated model weights.

The two newly supplied weights were checked with Ultralytics on Windows:
`yolo26l.pt` loaded as a detection model (80 classes) and
`yolo11m-pose.pt` loaded as a pose model (1 class). This does not verify their
Linux/GPU runtime behavior.

## Transfer and stage on Ubuntu

Transfer the archive and its checksum securely. Do not transfer `.env`,
`pythonai.json`, registered faces, production database data, or alert images
inside the release archive.

```bash
scp release/wave-vms-2026.10.22.tar.gz \
    release/wave-vms-2026.10.22.tar.gz.sha256 \
    release/wave-vms-2026.10.22-install.sh \
release/wave-vms-2026.10.22-install.sh.sha256 \
<vm-user>@<vm-address>:/tmp/
```

On the VM, first check its architecture/OS and validate the transferred
archive. `WAVE_SERVICE_USER` and `WAVE_SERVICE_GROUP` must be the existing
service account and group:

```bash
cd /tmp
sha256sum --check wave-vms-2026.10.22.tar.gz.sha256
sha256sum --check wave-vms-2026.10.22-install.sh.sha256
sudo WAVE_SERVICE_USER=<existing-user> WAVE_SERVICE_GROUP=<existing-group> \
  /tmp/wave-vms-2026.10.22-install.sh \
  /tmp/wave-vms-2026.10.22.tar.gz 2026.10.22 /var/lib/wave-vms-staging
```

The installer extracts to a new versioned directory. It refuses to overwrite
an existing version, leaves `current` and systemd untouched, copies only
missing model/config templates, and does not touch PostgreSQL. Edit
`/var/lib/wave-vms-staging/.env` and
`/var/lib/wave-vms-staging/config/cameras.json`; the generated camera template
is intentionally disabled and sanitized. Configure `ENABLED_CAMERAS` and the
required camera/NVR addresses in the external environment/configuration.
Provision the authorized
`pythonai.json` and `buffalo_l` cache outside the release. Preserve the
existing configuration/data and permissions; never point staging at
production data for write/alert tests.

The repository has separate service templates:
`packaging/wave-vms-source.service` and
`packaging/wave-vms-build.service`. Both use the same service account,
external `WAVE_RUNTIME_DIR`, port and persistent storage. Source mode requires
a deliberately deployed source checkout and `.venv`; compiled releases do
not contain Python source. Update the service account and source/build paths
for the target VM before installing either unit.

Install both templates but do not enable or start both at once:

```bash
sudo install -m 0644 packaging/wave-vms-source.service \
  /etc/systemd/system/wave-vms-source.service
sudo install -m 0644 packaging/wave-vms-build.service \
  /etc/systemd/system/wave-vms-build.service
sudo systemctl daemon-reload
sudo scripts/switch_mode.sh source
sudo systemctl status wave-vms-source.service --no-pager
sudo scripts/switch_mode.sh build
sudo systemctl status wave-vms-build.service --no-pager
sudo scripts/health_check.sh http://127.0.0.1:5000/api/cameras
```

The switch helper stops the other mode, verifies it stopped, starts the
selected service, and waits for the API health endpoint. If the selected mode
fails its health check, it stops that unit and reports how to inspect its
logs; it does not silently start the other mode. Inspect logs with
`sudo journalctl -u wave-vms-build.service -n 200 --no-pager` (or use the
source unit name). Leave automatic startup disabled for both services unless
boot configuration guarantees that only one mode is started.

Do not cut over until staging has verified API responses, Socket.IO connection
and event delivery, every intended camera worker, model loading, ONNX providers,
GPU inference, PostgreSQL behavior against a disposable database, face
registration/matching with authorized test data, notifications in safe test
mode, writable paths, and clean shutdown/restart. This workspace has not
executed those production/staging checks.

## Production cutover and rollback

Preserve the existing service, source tree, Nginx, frontend, and runtime data
until staging sign-off. The compiled service template uses
`/opt/wave-vms/current/wave-vms`; install and review it only when production
cutover is approved. Keep `WAVE_RUNTIME_DIR` pointed at the existing external
configuration and persistent-data directory.

Before switching, confirm the old backend responds at
`WAVE_HEALTH_URL` (defaults to `http://127.0.0.1:5000/api/cameras`). The
activation script saves the current symlink, atomically changes it, restarts
`wave-vms.service`, and rolls back the pointer if restart or HTTP health check
fails. It refuses to proceed unless the selected systemd unit already runs
`/opt/wave-vms/current/wave-vms`. Preserve the existing unit and source tree
until this reviewed service configuration is explicitly approved:

```bash
sudo WAVE_HEALTH_URL=http://127.0.0.1:5000/api/cameras \
  /opt/wave-vms/releases/wave-vms-2026.10.22/scripts/activate_release.sh \
  2026.10.22
```

Manual rollback:

```bash
sudo WAVE_HEALTH_URL=http://127.0.0.1:5000/api/cameras \
  /opt/wave-vms/current/scripts/rollback.sh
sudo journalctl -u wave-vms.service -n 200 --no-pager
```

The existing Nginx location, frontend `dist/` root, API proxy, and Socket.IO
upgrade headers must remain unchanged unless independently validated. Keep the
backend port private behind the existing Nginx policy.

## Configuration, data, and credentials

`WAVE_RUNTIME_DIR` is the external root. The application resolves:

```text
.env
config/cameras.json
models/*.pt
known_faces/
alert_images/
.insightface/
```

Relative model and Firebase credential paths resolve under this root (models
also resolve under its `models/` directory). Existing source-tree layouts keep
working. `FIREBASE_SERVICE_ACCOUNT` and `INSIGHTFACE_HOME` may be absolute
paths. Credentials are not copied to a release; create and permission the
external configuration yourself. Admin/user accounts are not bootstrapped from implicit default credentials;
set the corresponding username and password explicitly only when bootstrapping
is intended. Existing database accounts are not changed.

PostgreSQL settings are required by the active database helper. `PUBLIC_BASE_URL`
should be set to the externally reachable image URL; no development IP is
substituted. Use the existing database and service account only after
staging, and keep all credentials out of shell history and release archives.

## Verification and known limitations

```bash
/path/to/release/scripts/verify_release.sh /path/to/release
/path/to/release/scripts/health_check.sh http://127.0.0.1:5000/api/cameras
```

The release verifier checks hashes, Linux ELF format, architecture, shared
library resolution, source-file exclusion, and the default weights listed
above. The HTTP check proves only that `/api/cameras` responds. Neither proves
camera connectivity, GPU execution, Socket.IO events, database writes,
notifications, or operational readiness.

No Linux executable or archive was generated in this Windows workspace. The
target VM's Ubuntu/glibc/architecture, NVIDIA driver/CUDA stack, installed
system libraries, Nginx/frontend layout, production data locations, and
service-account name are unknown and must be measured before deployment.
