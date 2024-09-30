import sys
import contextlib


@contextlib.contextmanager
def multi_print(*files):
    original_stdout = sys.stdout
    sys.stdout = open(files[0], 'w') if len(files) == 1 else MultiStream(*files)
    try:
        yield
    finally:
        sys.stdout.close() if len(files) == 1 else None
        sys.stdout = original_stdout


class MultiStream:
    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for f in self.files:
            f.write(data)

    def flush(self):
        for f in self.files:
            f.flush()
