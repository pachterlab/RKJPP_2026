"""Linear--Gaussian recoverability model (compatibility shim).

The estimator now lives in the standalone ``attainable-information`` package
(https://github.com/pachterlab/attainable_information). This module re-exports
it so that ``rgit.model`` keeps working for the cohort pipeline, notebooks, and
scripts in this repository.
"""

from attainable_information.recoverability import *  # noqa: F401,F403
from attainable_information.recoverability import (  # noqa: F401
    __all__,
    _covariance,
    _double_centered_distance,
    _inv_sqrt,
)
