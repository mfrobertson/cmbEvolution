import numpy as np
from scipy import stats, optimize
import matplotlib.pyplot as plt


class ScaleInvariantPSM:
    """
    Calculates the values of the inflationary produced scale-invariant power spectrum at specific points in momentum
    space.

    Attributes
    ----------
    kM : ndarray
        2D array of k-values over which the power spectrum will be calculated at.
    As : int or float
        Amplitude of scalar fluctuations at the pivot scale.
    ns: float
        Scalar spectral index.
    kp: int or float
        Pivot scale.
    psM : ndarray
        2D array containing the value of the power spectrum at each point in the momentum space given by kM.
    """

    def __init__(self, kM, As=2.1e-9, ns=0.96, kp=0.05):
        """
        Constructor.

        Parameters
        ----------
        kM : ndarray
            2D array of k-values over which the power spectrum will be calculated at.
        As : int or float
            Amplitude of scalar fluctuations at the pivot scale.
        ns: float
            Scalar spectral index.
        kp: int or float
            Pivot scale.
        """
        self.kM = kM
        self.As = As
        self.ns = ns
        self.kp = kp
        self.psM = self._psMatrix()

    def _psMatrix(self):
        self.kM[0, 0] = 1  # Avoid division by 0 where possible
        with np.errstate(divide='ignore'):
            psM = 2*np.pi * self.kM**(-2) * self.As * (self.kM / self.kp) ** (self.ns - 1)
        self.kM[0, 0] = 0  # Resetting
        return psM


class CalculatePS:
    """
    Calculates the power spectrum of a given field over a number of k-bins.

    Attributes
    ----------
    fftField : Field
        Fourier transformed field.
    kM : ndarray
        2D array of the momentum-space k-values corresponding the field.
    kp :
        Pivot scale.
    ps : ndarray
        1D array of the calculated mean value of the power spectrum over each k-bin.
    psErrors : ndarray
        1D array of the error in the mean at each k-bin.
    kBins : ndarray
        1D array of the mode k-value in each k-bin.
    As : float
        Calculated inflationary parameter.
    ns : float
        Calculated inflationary parameter.
    paramErrors : list
        Length 2 list containing the fit error in each calculated inflationary parameter.
    """

    def __init__(self, field, raw=False, kp=0.05, bins=10):
        """
        Constructor.

        Parameters
        ----------
        field : Field
            Field from which to calculate power spectrum.
        raw : bool
            Calculate 'raw' power spectrum.
        kp : float
            Pivot scale.
        bins : int
            Number of k-bins.
        """
        self.fftField = field.FFT()
        self.kM = field.kMatrix()
        self.kp = kp
        self._microKelvinSqr = (2.725e6)**2
        self._distToLastScatter = 13900

        self.ps, self.psErrors, self.kBins = self._calculatePS(isRaw=raw, bins=bins)
        self.As, self.ns, self.paramErrors = self._getSIParams()

    def _calculatePS(self, isRaw, bins):
        ps2D_raw = np.abs(self.fftField.field)**2
        kFlat = self.kM.flatten()

        if isRaw:
            psFlat = (ps2D_raw).flatten()
        else:
            psFlat = (kFlat**2) * (ps2D_raw).flatten() * (2*np.pi)**-1

        means, bin_edges, binnumber = stats.binned_statistic(kFlat[1:], psFlat[1:], 'mean', bins=bins)
        counts, *others = stats.binned_statistic(kFlat[1:], psFlat[1:], 'count', bins=bins)
        stds, *others = stats.binned_statistic(kFlat[1:], psFlat[1:], 'std', bins=bins)
        errors = stds/np.sqrt(counts)

        binSeperation = bin_edges[1]
        kBins = np.asarray([bin_edges[i] - binSeperation/2 for i in range(1, len(bin_edges))])

        ps = np.asarray(means)

        return ps, errors, kBins

    def drawPS(self, title=None, siFit=False, units=False):
        """
        Quick Method for drawing Matplotlib plot of power spectrum.

        Parameters
        ----------
        title : str
            Plot title.
        siFit : bool
            Include fit of scale invariant parameters.
        units : bool
            Convert units to Kelvin squared, and ell.

        Returns
        -------

        """
        if units:
            ell = self.kBins * self._distToLastScatter
            ps = self.ps * self._microKelvinSqr
            psErrors = self.psErrors * self._microKelvinSqr

            plt.figure()
            plt.plot(ell, ps, ".")
            plt.errorbar(ell, ps, yerr=psErrors, ls="None")

            xlabel = "$l$"
            ylabel = r"$\dfrac{k^2}{2\pi}P(k)$ [$\mu$K$^2$]"

            spacings = np.linspace(0, ell[-1], 100)
            fitPS = self._siFitFunc_withUnits(spacings, self.As, self.ns - 1)
        else:
            plt.figure()
            plt.plot(self.kBins, self.ps, ".")
            plt.errorbar(self.kBins, self.ps, yerr=self.psErrors, ls="None")

            xlabel = ("$k$ [Mpc]$^{-1}$")
            ylabel = (r"$\dfrac{k^2}{2\pi}P(k)$")

            spacings = np.linspace(0, self.kBins[-1], 100)
            fitPS = self._siFitFunc(spacings, self.As, self.ns - 1)

        if siFit:
            plt.plot(spacings, fitPS)
            txtStr = f"$A_s = ${self.As:.3e}$\pm${self.paramErrors[0]:.1e}\n$k_p = ${self.kp}\n$n_s = ${self.ns:.4f}$\pm${self.paramErrors[1]:.1e}"
            plt.text(0.8, 0.85, txtStr, horizontalalignment='center',verticalalignment='center', transform=plt.gca().transAxes)

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.draw()

    def _siFitFunc_withUnits(self, k, a, b):
        return a * (k / (self.kp*self._distToLastScatter))**b * self._microKelvinSqr

    def _siFitFunc(self, k, a, b):
        return a * (k / self.kp)**b

    def _getSIParams(self):
        popt, pcov = optimize.curve_fit(self._siFitFunc, self.kBins, self.ps)
        As = popt[0]
        ns = popt[1] + 1
        err = np.sqrt(np.diag(pcov))
        return As, ns, err
