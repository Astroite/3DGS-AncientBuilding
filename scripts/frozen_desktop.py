"""PyInstaller entry point for the Windows workbench."""
from gsstudio.infrastructure.runtime.layout import configure_release_path
from gsstudio.interfaces.desktop.app import main

if __name__ == "__main__":
    configure_release_path()
    raise SystemExit(main())
