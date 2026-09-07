"""Application entry point.

The existing pipeline remains in alcon_stream_new.py during the staged
organization refactor. Keeping this launcher small preserves current startup
behavior while providing the new stable command: python app.py.
"""

from alcon_stream_new import main


if __name__ == "__main__":
    main()
