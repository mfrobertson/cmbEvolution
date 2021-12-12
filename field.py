import numpy as np
from scipy import fft
import matplotlib.pyplot as plt


class Field:
    """
    2D field.

    Attributes
    ----------
    field : ndarray
        2D square real valued array.
    N : int
        Number of pixels of array.
    scale : int or float
        Physical length of field [Mpc].
    space : str
        "fourier" or "physical".
    """

    def __init__(self, field, N, scale, space):
        """
        Constructor.

        Parameters
        ----------
        field : ndarray
            2D square real valued array.
        N : int
            Number of pixels of array.
        scale : int or float
            Physical length of field [Mpc].
        space : str
            "fourier" or "physical".
        """
        self.field = field
        self.N = N
        self.scale = scale
        self.space = space

    def FFT(self):
        """
        Fourier transforms the field.

        Returns
        -------
        Field
            The Fourier transformed field.
        """
        fieldFFT = fft.rfftn(self.field, norm="ortho")
        return Field(fieldFFT, self.N, self.scale, "fourier")

    def iFFT(self):
        """
        Inverse Fourier transforms the field.

        Returns
        -------
        Field
            The Inverse Fourier transformed field.
        """
        field = fft.irfftn(self.field, norm="ortho")
        return Field(field, self.N, self.scale, "physical")

    def kMatrix(self):
        """
        Generates 2D array of the k-values corresponding to this Field.

        Returns
        -------
        ndarray
            2D array of k-values.
        """
        return Kmatrix(self.N, self.scale).kM

    def sampleKs(self, N_perLog=200):
        """
        Calculate appropriate sample of k-values for this Field, to be used as input for CAMB transfer functions.

        Parameters
        ----------
        N_perLog : int
            Number of samples per logarithmic interval.

        Returns
        -------
        ndarray
            1D array of sampled k-values.
        """
        kmin = 2 * np.pi / self.scale
        kmax = np.pi * self.N / self.scale
        kmin_round = np.int(np.floor(np.log10(kmin)))
        kmax_round = np.int(np.ceil(np.log10(kmax)))
        Nsamples = (kmax_round - kmin_round) * N_perLog
        return np.logspace(kmin_round, kmax_round, Nsamples)

    def drawField(self, title=None, clims=None, cbar=True, units=False):
        """
        Quick method for drawing Matplotlib plot field.

        Parameters
        ----------
        title : str
            Plot title.
        clims : 2-tuple
            Colorbar limits.
        cbar : bool
            Display the colorbar.
        units : bool
            If True, coverts units of physical field to Kelvin, and for Fourier field converts axis units to ell values.

        Returns
        -------
        None

        """
        if self.space == "physical":
            extent = [0, self.scale, 0, self.scale]
            if units:
                microKelvin = 2.725e6
                field = self.field * microKelvin
                cbTitle = "$\Delta\mu$K"
            else:
                field = self.field
                cbTitle = None
            xlabel = "$x$ [Mpc]"
            ylabel = "$y$ [Mpc]"

        else:
            if units:
                maximum = 13900 * np.pi * self.N / self.scale
                xlabel = "$l_x$"
                ylabel = "$l_y$"
            else:
                maximum = np.pi * self.N / self.scale
                xlabel = "$k_x$ [Mpc$^{-1}$]"
                ylabel = "$k_y$ [Mpc$^{-1}$]"

            extent = [0, maximum, maximum, 0]
            field = np.abs(self.field[:self.N // 2, :])
            cbTitle = None

        plt.figure()
        plt.imshow(field, extent=extent, cmap="jet")
        cb = plt.colorbar()
        cb.ax.set_title(cbTitle)
        if clims is not None:
            plt.clim(clims[0], clims[1])

        if not cbar:
            cb.remove()

        if title is not None:
            plt.title(title)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.draw()


class Kmatrix:
    """
    Calculates momentum-space k-values corresponding to a 2D real square array.

    Attributes
    ----------
    N : int
        Number of pixels of real array.
    scale : int or float
        Physical length of field [Mpc].
    kM : ndarray
        2D array of calculated k-values.
    """

    def __init__(self, N, scale):
        """
        Constructor.

        Parameters
        ----------
        N : int
            Number of pixels of real array.
        scale : int or float
            Physical length of field [Mpc].
        """
        self.N = N
        self.scale = scale
        self.kM = self._kMatrix()

    def _kMatrix(self):
        kx = fft.rfftfreq(self.N, d=self.scale / self.N) * 2 * np.pi
        ky = fft.fftfreq(self.N, d=self.scale / self.N) * 2 * np.pi
        kSqr = kx[np.newaxis, ...] ** 2 + ky[..., np.newaxis] ** 2
        return np.sqrt(kSqr)
