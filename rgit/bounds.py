"""Attainable-information bounds (compatibility shim).

The bounds now live in the standalone ``attainable-information`` package
(https://github.com/pachterlab/attainable_information). This module re-exports
them so that ``rgit.bounds`` keeps working for the scripts in this repository.
"""

from attainable_information.bounds import *  # noqa: F401,F403
from attainable_information.bounds import __all__, _LOG2, _as_rho2  # noqa: F401
