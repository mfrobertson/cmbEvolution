import numpy as np
from scipy import stats, optimize
import matplotlib.pyplot as plt

class GeneratePS:

    def __init__(self, kMatrix, As=2.1e-9, ns=0.96, kp=0.05):
        self.kM = kMatrix
        self.As = As
        self.ns = ns
        self.kp = kp
        self.psM = self.psMatrix()

    def psMatrix(self):
        self.kM[0, 0] = 1  # Avoid division by 0 where possible
        with np.errstate(divide='ignore'):
            psM = 2*np.pi**2 * self.kM**(-2) * self.As * (self.kM / self.kp) ** (self.ns - 1)
        self.kM[0, 0] = 0  # Resetting
        return psM


class CalculatePS:

    def __init__(self, field, raw=False, kp=0.05, bins=10):
        self.fftField = field.FFT()
        self.kMatrix = field.kMatrix()
        self.kp = kp
        self.microKelvinConv = (2.72e6)**2
        self.ellConv = 13900

        self.ps, self.psErrors, self.kBins = self.calculatePS(isRaw=raw, bins=bins)
        self.As, self.ns, self.paramErrors = self.getSIParams()

    def calculatePS(self, isRaw, bins):
        ps2D_raw = np.abs(self.fftField.field)**2
        kFlat = self.kMatrix.flatten()

        if isRaw:
            psFlat = (ps2D_raw).flatten()
        else:
            psFlat = (kFlat**2) * (ps2D_raw).flatten() * (2*np.pi**2)**-1

        means, bin_edges, binnumber = stats.binned_statistic(kFlat[1:], psFlat[1:], 'mean', bins=bins)
        counts, *others = stats.binned_statistic(kFlat[1:], psFlat[1:], 'count', bins=bins)
        stds, *others = stats.binned_statistic(kFlat[1:], psFlat[1:], 'std', bins=bins)
        errors = stds/np.sqrt(counts)

        binSeperation = bin_edges[1]
        kBins = np.asarray([bin_edges[i] - binSeperation/2 for i in range(1, len(bin_edges))])

        ps = np.asarray(means)

        return ps, errors, kBins

    def drawPS(self, title=None, siFit=False, units=False):
        if units:
            ell = self.kBins * self.ellConv
            ps = self.ps * self.microKelvinConv
            psErrors = self.psErrors * (2.72e6)**2

            plt.figure()
            plt.plot(ell, ps, ".")
            plt.errorbar(ell, ps, yerr=psErrors, ls="None")

            xlabel = "$l$"
            ylabel = r"$\dfrac{k^2}{2\pi^2}P(k)$ [$\mu$K$^2$]"

            spacings = np.linspace(0, ell[-1], 100)
            fitPS = self.siFitFunc_withUnits(spacings, self.As, self.ns - 1)
        else:
            plt.figure()
            plt.plot(self.kBins, self.ps, ".")
            plt.errorbar(self.kBins, self.ps, yerr=self.psErrors, ls="None")

            xlabel = ("$k$ [Mpc]$^{-1}$")
            ylabel = (r"$\dfrac{k^2}{2\pi^2}P(k)$")

            spacings = np.linspace(0, self.kBins[-1], 100)
            fitPS = self.siFitFunc(spacings, self.As, self.ns - 1)

        if siFit:
            plt.plot(spacings, fitPS)
            txtStr = f"$A_s = ${self.As:.3e}$\pm${self.paramErrors[0]:.1e}\n$k_p = ${self.kp}\n$n_s = ${self.ns:.4f}$\pm${self.paramErrors[1]:.1e}"
            plt.text(0.8, 0.85, txtStr, horizontalalignment='center',verticalalignment='center', transform=plt.gca().transAxes)

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.draw()

    def siFitFunc_withUnits(self, k, a, b):
        return a * (k / (self.kp*self.ellConv))**b * self.microKelvinConv

    def siFitFunc(self, k, a, b):
        return a * (k / self.kp)**b

    def getSIParams(self):
        popt, pcov = optimize.curve_fit(self.siFitFunc, self.kBins, self.ps)
        As = popt[0]
        ns = popt[1] + 1
        err = np.sqrt(np.diag(pcov))
        return As, ns, err
