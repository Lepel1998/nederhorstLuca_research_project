import os
import shutil


def IgnoreFiles(dir, files):
    return [f for f in files if os.path.isfile(os.path.join(dir, f))]
    