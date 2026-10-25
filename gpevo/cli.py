"""Console entry point for the packaged ``gpevo`` application."""
from __future__ import annotations

def main(argv=None):
    from gpevo.FunctionFolder.USPEX.Start import Start
    Start()


if __name__ == "__main__":
    main()
