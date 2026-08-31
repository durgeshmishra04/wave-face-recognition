"""
Test script for POST /api/register-person
==========================================
Uses existing images from known_faces/Kapil Kumar as test data.

Run:
    python test_register_person.py
"""

import os
import sys
import json
import requests
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIG — edit if your server runs elsewhere
# ─────────────────────────────────────────────
BASE_URL = "http://localhost:5000"
ENDPOINT = f"{BASE_URL}/api/register-person"

# Test person details (change freely)
TEST_PAYLOAD = {
    "gate_no":       "Gate 1",
    "employee_name": "Test Employee API",
    "designation":   "QA Engineer",
    "employee_id":   "TEST_EMP_999",  # must be unique; change to re-run
}

# Grab 5 images from an existing known person folder
SCRIPT_DIR   = Path(__file__).resolve().parent
KNOWN_FACES  = SCRIPT_DIR / "known_faces"
IMAGE_SOURCE = KNOWN_FACES / "Kapil Kumar"   # folder with 5 images


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"


def ok(msg):   print(f"{GREEN}  [PASS]{RESET} {msg}")
def fail(msg): print(f"{RED}  [FAIL]{RESET} {msg}")
def info(msg): print(f"{CYAN}  [INFO]{RESET} {msg}")
def warn(msg): print(f"{YELLOW}  [WARN]{RESET} {msg}")


def collect_images(folder, count=5):
    """Return exactly `count` image files from folder."""
    supported = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in supported
    )
    if len(images) < count:
        print(f"{RED}ERROR:{RESET} Need at least {count} images in {folder}, "
              f"found {len(images)}.")
        sys.exit(1)
    return images[:count]


# ─────────────────────────────────────────────
# TEST CASES
# ─────────────────────────────────────────────

def test_server_reachable():
    """Connectivity pre-check before running other tests."""
    print(f"\n{YELLOW}>> Pre-check: Server reachable at {BASE_URL}{RESET}")
    try:
        r = requests.get(f"{BASE_URL}/api/status", timeout=5)
        if r.ok:
            ok(f"Server is up -- {r.json()}")
            return True
        else:
            fail(f"Server responded with {r.status_code}")
    except requests.exceptions.ConnectionError:
        fail(f"Cannot connect to {BASE_URL}. Is the Flask server running?")
    except requests.exceptions.Timeout:
        fail("Connection timed out.")
    return False


def test_missing_field():
    """400 when a required field (employee_name) is omitted."""
    print(f"\n{YELLOW}>> Test 1: Missing required field (employee_name){RESET}")
    images = collect_images(IMAGE_SOURCE)
    files  = {f"image{i+1}": open(str(p), "rb") for i, p in enumerate(images)}
    data   = {k: v for k, v in TEST_PAYLOAD.items() if k != "employee_name"}

    try:
        r = requests.post(ENDPOINT, data=data, files=files, timeout=30)
        if r.status_code == 400:
            ok(f"Got 400 -- {r.json().get('message')}")
        else:
            fail(f"Expected 400, got {r.status_code}: {r.text[:200]}")
    finally:
        for f in files.values():
            f.close()


def test_wrong_image_count():
    """400 when fewer than 5 images are sent."""
    print(f"\n{YELLOW}>> Test 2: Wrong image count (only 3 images){RESET}")
    images = collect_images(IMAGE_SOURCE)[:3]
    files  = {f"image{i+1}": open(str(p), "rb") for i, p in enumerate(images)}

    try:
        r = requests.post(ENDPOINT, data=TEST_PAYLOAD, files=files, timeout=30)
        if r.status_code == 400:
            ok(f"Got 400 -- {r.json().get('message')}")
        else:
            fail(f"Expected 400, got {r.status_code}: {r.text[:200]}")
    finally:
        for f in files.values():
            f.close()


def test_successful_registration():
    """201 on a valid registration with 5 face images."""
    print(f"\n{YELLOW}>> Test 3: Successful registration{RESET}")
    images = collect_images(IMAGE_SOURCE)
    files  = {f"image{i+1}": open(str(p), "rb") for i, p in enumerate(images)}

    info(f"Endpoint : {ENDPOINT}")
    info(f"Payload  : {json.dumps(TEST_PAYLOAD)}")
    info(f"Images   : {[p.name for p in images]}")

    try:
        r = requests.post(ENDPOINT, data=TEST_PAYLOAD, files=files, timeout=60)
        info(f"Status   : {r.status_code}")

        try:
            body = r.json()
            info(f"Response :\n{json.dumps(body, indent=4)}")
        except Exception:
            info(f"Raw body : {r.text[:300]}")
            body = {}

        if r.status_code == 201 and body.get("success"):
            ok("Employee registered successfully!")
            ok(f"registration_id = {body.get('registration_id')}")
        elif r.status_code == 400:
            warn(f"400 Bad Request -- {body.get('message')}")
            warn("Tip: If 'Employee ID already registered', change TEST_PAYLOAD employee_id and re-run.")
        elif r.status_code == 503:
            warn(f"503 Service Unavailable -- {body.get('message')}")
            warn("Tip: Make sure the Flask server is running and face model is loaded.")
        else:
            fail(f"Unexpected {r.status_code}: {r.text[:300]}")
    finally:
        for f in files.values():
            f.close()


def test_duplicate_employee_id():
    """400 when the same employee_id is registered twice."""
    print(f"\n{YELLOW}>> Test 4: Duplicate employee_id (re-registering same ID){RESET}")
    images = collect_images(IMAGE_SOURCE)
    files  = {f"image{i+1}": open(str(p), "rb") for i, p in enumerate(images)}

    try:
        r = requests.post(ENDPOINT, data=TEST_PAYLOAD, files=files, timeout=60)
        if r.status_code == 400:
            ok(f"Got 400 on duplicate -- {r.json().get('message')}")
        elif r.status_code == 201:
            warn("Got 201 -- first run succeeded (Test 3 was likely skipped). "
                 "Re-run the script to trigger the duplicate check.")
        else:
            fail(f"Unexpected {r.status_code}: {r.text[:200]}")
    finally:
        for f in files.values():
            f.close()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  API TEST: POST /api/register-person")
    print("=" * 55)

    if not IMAGE_SOURCE.is_dir():
        print(f"{RED}ERROR:{RESET} Image source folder not found: {IMAGE_SOURCE}")
        sys.exit(1)

    if not test_server_reachable():
        print(f"\n{RED}Aborting -- server is not reachable.{RESET}")
        sys.exit(1)

    test_missing_field()
    test_wrong_image_count()
    test_successful_registration()
    test_duplicate_employee_id()

    print(f"\n{'=' * 55}")
    print("  All tests complete.")
    print("=" * 55)
