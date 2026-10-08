"""Optional command-line and doctor entry point in the offline release."""
from gsstudio.infrastructure.runtime.layout import configure_release_path
from gsstudio.interfaces.cli.app import app

if __name__ == "__main__":
    configure_release_path()
    app()
