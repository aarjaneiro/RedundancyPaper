import warnings

import numpy as np
from pandas.util._decorators import deprecate_kwarg
from scipy import stats
from statsmodels.compat import lzip
from statsmodels.tools.validation import bool_like, int_like, float_like, string_like, array_like
from statsmodels.tsa.stattools import q_stat, acovf


@deprecate_kwarg("unbiased", "adjusted")
def acf_mc(
    x,
    samp_size,
    std,
    adjusted=False,
    nlags=None,
    qstat=False,
    fft=None,
    alpha=None,
    missing="none",
):
    """
Same as acf but for samp_size runs.
    """
    adjusted = bool_like(adjusted, "adjusted")
    nlags = int_like(nlags, "nlags", optional=True)
    qstat = bool_like(qstat, "qstat")
    fft = bool_like(fft, "fft", optional=True)
    alpha = float_like(alpha, "alpha", optional=True)
    missing = string_like(
        missing, "missing", options=("none", "raise", "conservative", "drop")
    )
    if nlags is None:
        warnings.warn(
            "The default number of lags is changing from 40 to"
            "min(int(10 * np.log10(nobs)), nobs - 1) after 0.12"
            "is released. Set the number of lags to an integer to "
            " silence this warning.",
            FutureWarning,
        )
        nlags = 40

    if fft is None:
        warnings.warn(
            "fft=True will become the default after the release of the 0.12 "
            "release of statsmodels. To suppress this warning, explicitly "
            "set fft=False.",
            FutureWarning,
        )
        fft = False
    x = array_like(x, "x")
    nobs = len(x)  # TODO: should this shrink for missing="drop" and NaNs in x?
    avf = acovf(x, adjusted=adjusted, demean=True, fft=fft, missing=missing)
    acf = avf[: nlags + 1] / avf[0]
    if not (qstat or alpha):
        return acf
    if alpha is not None:
        adj = std/np.sqrt(samp_size)
        varacf = np.ones_like(acf) / nobs
        varacf[0] = 0
        varacf[1] = 1.0 / nobs
        varacf[2:] *= 1 + 2 * np.cumsum(acf[1:-1] ** 2)
        interval = stats.norm.ppf(1 - alpha / 2.0) * np.sqrt(varacf)
        confint = np.array(lzip(acf - interval * adj, acf + interval * adj))
        if not qstat:
            return acf, confint
    if qstat:
        qstat, pvalue = q_stat(acf[1:], nobs=nobs)  # drop lag 0
        if alpha is not None:
            return acf, confint, qstat, pvalue
        else:
            return acf, qstat, pvalue