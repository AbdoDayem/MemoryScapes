import sys

from memoryscapes_workers.cli import main

if __name__ == "__main__":  # required: stages run in spawned processes that re-import this
    sys.exit(main())
