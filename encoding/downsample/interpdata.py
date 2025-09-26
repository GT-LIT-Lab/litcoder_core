"""Interpolation utilities for downsampling continuous data."""

import numpy as np
import logging

logger = logging.getLogger("text.regression.interpdata")


def interpdata(data, oldtime, newtime):
    """Interpolates the columns of [data] to find the values at [newtime], given that the current
    values are at [oldtime].  [oldtime] must have the same number of elements as [data] has rows.
    """
    if not len(oldtime) == data.shape[0]:
        raise IndexError("oldtime must have same number of elements as data has rows.")

    newdata = np.empty((len(newtime), data.shape[1]))

    for ci in range(data.shape[1]):
        if (ci % 100) == 0:
            logger.info("Interpolating column %d/%d.." % (ci + 1, data.shape[1]))

        newdata[:, ci] = np.interp(newtime, oldtime, data[:, ci])

    return newdata


def sincfun(B, t, window=np.inf, causal=False, renorm=True):
    """Compute the sinc function with some cutoff frequency [B] at some time [t]."""
    val = 2 * B * np.sin(2 * np.pi * B * t) / (2 * np.pi * B * t + 1e-20)
    if t.shape:
        val[np.abs(t) > window / (2 * B)] = 0
        if causal:
            val[t < 0] = 0
        if not np.sum(val) == 0.0 and renorm:
            val = val / np.sum(val)
    elif np.abs(t) > window / (2 * B):
        val = 0
        if causal and t < 0:
            val = 0
    return val


def lanczosfun(cutoff, t, window=3):
    """Compute the lanczos function with some cutoff frequency [B] at some time [t].
    [t] can be a scalar or any shaped numpy array.
    If given a [window], only the lowest-order [window] lobes of the sinc function
    will be non-zero.

    Args:
        cutoff: Cutoff frequency
        t: Time points
        window: Number of lobes in the window

    Returns:
        Lanczos function values
    """
    t = t * cutoff
    val = window * np.sin(np.pi * t) * np.sin(np.pi * t / window) / (np.pi**2 * t**2)
    val[t == 0] = 1.0
    val[np.abs(t) > window] = 0.0
    return val


def sincinterp2D(
    data, oldtime, newtime, cutoff_mult=1.0, window=1, causal=False, renorm=True
):
    """Interpolates the columns of [data] using sinc interpolation."""
    # log the cutoff multiplier # TODO: remove later.
    logger.info(
        f"Doing sinc interpolation with cutoff multiplier={cutoff_mult} and {window} lobes."
    )
    cutoff = 1 / np.mean(np.diff(newtime)) * cutoff_mult
    print("Doing sinc interpolation with cutoff=%0.3f and %d lobes." % (cutoff, window))

    sincmat = np.zeros((len(newtime), len(oldtime)))
    for ndi in range(len(newtime)):
        sincmat[ndi, :] = sincfun(
            cutoff, newtime[ndi] - oldtime, window, causal, renorm
        )

    newdata = np.dot(sincmat, data)
    return newdata


def lanczosinterp2D(data, oldtime, newtime, window=3, cutoff_mult=1.0, rectify=False):
    """Interpolates the columns of [data], assuming that the i'th row of data corresponds to
    oldtime(i). A new matrix with the same number of columns and a number of rows given
    by the length of [newtime] is returned.

    The time points in [newtime] are assumed to be evenly spaced, and their frequency will
    be used to calculate the low-pass cutoff of the interpolation filter.

    Args:
        data: Input data matrix
        oldtime: Original timestamps
        newtime: Target timestamps
        window: Number of lobes in the Lanczos window
        cutoff_mult: Multiplier for the cutoff frequency
        rectify: Whether to split positive and negative values

    Returns:
        Interpolated data matrix
    """
    ## Find the cutoff frequency ##
    cutoff = 1 / np.mean(np.diff(newtime)) * cutoff_mult

    ## Build up sinc matrix ##
    sincmat = np.zeros((len(newtime), len(oldtime)))

    for ndi in range(len(newtime)):
        sincmat[ndi, :] = lanczosfun(cutoff, newtime[ndi] - oldtime, window)

    if rectify:
        newdata = np.hstack(
            [
                np.dot(sincmat, np.clip(data, -np.inf, 0)),
                np.dot(sincmat, np.clip(data, 0, np.inf)),
            ]
        )
    else:
        # Construct new signal by multiplying the sinc matrix by the data ##
        newdata = np.dot(sincmat, data)

    return newdata


def gabor_xfm(data, oldtimes, newtimes, freqs, sigma):
    """Compute Gabor transform."""
    sinvals = np.vstack([np.sin(oldtimes * f * 2 * np.pi) for f in freqs])
    cosvals = np.vstack([np.cos(oldtimes * f * 2 * np.pi) for f in freqs])
    outvals = np.zeros((len(newtimes), len(freqs)), dtype=np.complex128)
    for ti, t in enumerate(newtimes):
        gaussvals = np.exp(-0.5 * (oldtimes - t) ** 2 / (2 * sigma**2)) * data
        sprod = np.dot(sinvals, gaussvals)
        cprod = np.dot(cosvals, gaussvals)
        outvals[ti, :] = cprod + 1j * sprod

    return outvals


def gabor_xfm2D(data, oldtimes, newtimes, freqs, sigma):
    """Compute 2D Gabor transform."""
    return np.vstack([gabor_xfm(d, oldtimes, newtimes, freqs, sigma).T for d in data])
