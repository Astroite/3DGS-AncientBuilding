"""PyInstaller entry point for the isolated pinned GPU runtime."""
import sys

from gsstudio.infrastructure.runtime.layout import configure_release_path


def main() -> int:
    configure_release_path()
    if sys.argv[1:] == ["--doctor"]:
        from gsstudio.interfaces.trainer_diagnostic import main as diagnose

        return diagnose()
    from gsstudio.pipeline.training.native import main as train

    train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
