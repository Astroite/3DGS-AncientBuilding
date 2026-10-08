"""PyInstaller entry point for one JSON-lines operation worker."""
from gsstudio.infrastructure.runtime.layout import configure_release_path
from gsstudio.interfaces.worker.main import main

if __name__ == "__main__":
    configure_release_path()
    raise SystemExit(main())
