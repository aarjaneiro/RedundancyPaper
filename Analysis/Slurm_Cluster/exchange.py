"""
Local exchangeability helpers

https://github.com/trevorcampbell/localexch
esp. https://github.com/trevorcampbell/localexch/blob/master/examples/crashdata/crash_analysis.ipynb
"""
# Imported separately because of numpy c api import issues
# https://stackoverflow.com/questions/14657375/cython-fatal-error-numpy-arrayobject-h-no-such-file-or-directory/14657667#14657667

import pyximport
import numpy as np

pyximport.install(setup_args={"include_dirs": np.get_include()}, reload_support=True)
# noinspection PyUnresolvedReferences
from _exchange import *

if __name__ == "__main__":
    import _exchange as ex
    print("\n-- Testing pyximports -- \n")
    for imp in dir(ex):
        if f"{imp}" not in ["np", "warn"] and "__" not in f"{imp}":
            print(f"{imp} imported.\n")
    print("-- Successfully pyximports --")
