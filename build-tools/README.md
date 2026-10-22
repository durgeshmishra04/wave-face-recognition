# Isolated Linux PyInstaller release

This directory contains build and deployment tooling only. It does not move,
rewrite, or delete application files. The spec explicitly builds the existing
`../app.py`; all generated build products default to the sibling directory
`../wave-vms-compiled/`, outside the source checkout.

The current development machine is Windows and has no configured compatible
Linux build environment. It cannot produce the requested Ubuntu executable.
Run these commands on Ubuntu matching the production VM's Ubuntu release,
architecture, glibc, and Python 3.11. Do not claim a release exists until the
build and staging tests complete.

## Build and package

From the source checkout root, first identify the target values on the VM:

```bash
uname -m
lsb_release -rs
getconf GNU_LIBC_VERSION
```

On a matching Ubuntu host, install Python 3.11, its venv support, and common
native build/runtime libraries (adjust package names for the target release):

```bash
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv python3.11-dev \
  build-essential pkg-config libgl1 libglib2.0-0 libgomp1 ffmpeg
```

Then run:

```bash
TARGET_UBUNTU_RELEASE=22.04 \
TARGET_ARCH=x86_64 \
TARGET_GLIBC_VERSION="glibc 2.35" \
bash build-tools/build_linux.sh 1.0.0
```

Replace the three target values with the VM's actual values. Optional
`WAVE_BUILD_OUTPUT=/absolute/path` changes the sibling output location.
Optional `PIP_CONSTRAINT_FILE=/path/to/constraints.txt` pins dependency
resolution. The build script refuses non-Linux, non-Ubuntu, target mismatch,
missing dependencies, incomplete PyInstaller output, and unresolved executable
libraries. It writes the venv, PyInstaller work, executable, lockfile, and
build metadata only under the external output directory.

Expected deliverables:

```text
../wave-vms-compiled/
  .build/                         # PyInstaller work and metadata
  .build-venv/                    # Dedicated build environment
  wave-vms-1.0.0/                 # Source-free, onedir release
  wave-vms-1.0.0-ubuntu.tar.gz
  wave-vms-1.0.0-ubuntu.tar.gz.sha256
```

The release contains the executable, PyInstaller runtime libraries, available
YOLO weights, build/dependency metadata, sanitized examples, installer and
verification scripts, and a SHA-256 file manifest. It excludes Python source,
`.env`, Firebase credentials, live camera configuration, face registrations,
databases, and persistent event data. `buffalo_l` is not in this repository;
provision the existing InsightFace cache separately on the VM. Model weights
are copied only if present in the source checkout; inspect
`models/ASSET_MANIFEST.txt` before deployment.

## Transfer to the VM

Run on the build host, substituting the actual host and destination:

```bash
scp ../wave-vms-compiled/wave-vms-1.0.0-ubuntu.tar.gz \
  ../wave-vms-compiled/wave-vms-1.0.0-ubuntu.tar.gz.sha256 \
  deploy-user@vm-host:/tmp/
```

On the VM:

```bash
cd /tmp
sha256sum --check wave-vms-1.0.0-ubuntu.tar.gz.sha256
tar -xzf wave-vms-1.0.0-ubuntu.tar.gz
```

## Install without switching production

The installer creates a versioned directory and missing runtime directories;
it does not enable, stop, restart, or replace a service and never overwrites
existing runtime files.

```bash
sudo WAVE_SERVICE_USER=existing-service-user \
  WAVE_SERVICE_GROUP=existing-service-group \
  bash wave-vms-1.0.0/scripts/install_release.sh \
  "$PWD/wave-vms-1.0.0-ubuntu.tar.gz" 1.0.0 /var/lib/wave-vms
```

Use the actual service account and existing persistent runtime path. If the
VM already has `.env`, `config/cameras.json`, models, face data, or other
persistent files, preserve them and verify the executable points to that same
runtime root. The installer seeds only absent example config/model files. It
does not copy or migrate a database.

The executable loads credentials and settings from `$RUNTIME_DIR/.env`.
Configure a separate compiled-mode unit as an explicit step. This command
creates a new unit only; it refuses to overwrite any existing unit, does not
enable or start it, and leaves the current service untouched:

```bash
sudo bash /opt/wave-vms/releases/wave-vms-1.0.0/scripts/configure_service.sh \
  1.0.0 existing-service-user existing-service-group /var/lib/wave-vms
sudo systemctl cat wave-vms-build.service
```

The generated unit uses the selected service account and runtime root. Review
it for deployment-specific port, environment overrides, and writable paths.
Do not overwrite the existing unit or start the new unit as part of installation.

Before any production switch, test the versioned binary independently in a
staging VM with a test database and safe notifications. Do not run source and
compiled processes at once against the same runtime, database writers, or
port. The shared runtime lock is a further guard, not a substitute for
stopping and verifying the prior service.

## Staging, switch, and rollback

Verify the unpacked package and inspect native dependencies:

```bash
wave-vms-1.0.0/scripts/verify_release.sh \
  /opt/wave-vms/releases/wave-vms-1.0.0
ldd /opt/wave-vms/releases/wave-vms-1.0.0/wave-vms
```

On staging, start the compiled release using the staging service's actual
configuration and check the expected API endpoint:

```bash
curl --fail --show-error http://127.0.0.1:5000/api/cameras
```

Also verify Socket.IO, PostgreSQL read/write behavior against test data,
YOLO/InsightFace loading and ONNX provider selection, camera reconnection,
face registration/recognition, ROI events/snapshots, notifications, service
shutdown/restart, and writable runtime paths. GPU, camera and integration
checks require the matching VM/devices/services; record them as NOT RUN or
REQUIRES PRODUCTION VM until actually performed.

Only after approval, switch services manually in a maintenance window:

```bash
sudo systemctl stop EXISTING_WAVE_SERVICE
sudo systemctl status EXISTING_WAVE_SERVICE --no-pager
sudo systemctl start wave-vms-build.service
sudo systemctl status wave-vms-build.service --no-pager
wave-vms-1.0.0/scripts/health_check.sh
```

Do not enable both source and compiled units on the same port. To roll back,
stop the compiled unit, confirm it stopped, restore/start the previously
recorded service/unit and executable path, then check that service's status
and health endpoint:

```bash
sudo systemctl stop wave-vms-build.service
sudo systemctl status wave-vms-build.service --no-pager
sudo systemctl start EXISTING_WAVE_SERVICE
sudo systemctl status EXISTING_WAVE_SERVICE --no-pager
curl --fail --show-error http://127.0.0.1:5000/api/cameras
```

Do not remove the source checkout as part of this workflow. Keep it securely
backed up so future Linux builds and deliberate source-enabled deployments
remain possible.
