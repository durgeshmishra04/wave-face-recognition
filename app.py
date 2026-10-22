"""Application entry point.

The existing pipeline remains in alcon_stream_new.py during the staged
organization refactor. Keeping this launcher small preserves current startup
behavior while providing the new stable command: python app.py.
"""

from dotenv import load_dotenv

from runtime_paths import acquire_runtime_lock, runtime_root

RUNTIME_ROOT = runtime_root()
load_dotenv(RUNTIME_ROOT / ".env")
_runtime_lock = acquire_runtime_lock()

from alcon_stream_new import main


if __name__ == "__main__":
    main()
