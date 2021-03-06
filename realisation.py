import copy
import numpy as np
from scipy import interpolate
from transfers import TransferFuncs
from powerspectrum import ScaleInvariantPSM
from field import Field, Kmatrix


class FieldRealisation:
    """
    Wrapper class for generating a realisation of 2D inflation produced scale invariant Fields, and evolving them with
    CAMB transfer functions.

    Attributes
    ----------
    N : int
        Number of pixels of array.
    scale : int or float
        Physical length of field [Mpc].
    kM : ndarray
        2D array of k-values.
    fftField : Field
        The momentum-space field.
    siField : Field
        The scale-invariant field.
    realField : Field
        The physical field.
    """

    def __init__(self, N=1024, scale=3000):
        """
        Constructor.

        Parameters
        ----------
        N : int
            Number of pixels of array.
        scale : int or float
            Physical length of field [Mpc].
        """
        self.N = N
        self.scale = scale

        self.kM = self.kMatrix()

        self._sigM = None
        self.fftField = None
        self.siField = None
        self.realField = None

    def buildSI(self, As=2.1e-9, ns=0.96, kp=0.05):
        """
        Generates a scale invariant field based off the Power Spectrum expected from cosmic inflation. A deepcopy of the
        physical field generated is stored at self.siField.

        Parameters
        ----------
        As : int or float
            Amplitude of scalar fluctuations at the pivot scale.
        ns: float
            Scalar spectral index.
        kp: int or float
            Pivot scale.

        Returns
        -------
        None
        """
        self._sigM = self._sigMatrix(As, ns, kp)
        self.fftField = Field(self._fftMatrix(), self.N, self.scale, "fourier")
        self.realField = self.fftField.iFFT()
        self.siField = copy.deepcopy(self.realField)

    def calcField(self, eta, source="monopole", tf_raw=None):
        """
        Evolves the field. If no transfer function is given, one will be calculated for a given eta and source.

        Parameters
        ----------
        eta : int
            Conformal time.
        source : str
            CAMB transfer function name.
        tf_raw : TransferFuncs
            Un-interpolated pre-calculated transfer function.

        Returns
        -------
        None
        """
        if self.siField is None:
            self.buildSI()

        sample_ks = self.siField.sampleKs()
        if tf_raw is None:
            tf_raw = TransferFuncs(sample_ks, eta, source).tfs[:, 0, 0]
        spl = interpolate.InterpolatedUnivariateSpline(sample_ks, tf_raw)
        tf = spl(self.kM)

        self.fftField = self.siField.FFT()
        self.fftField.field *= tf
        self.realField = self.fftField.iFFT()

    def kMatrix(self):
        """
        Generates 2D array of the k-values corresponding to these Fields.

        Returns
        -------
        ndarray
            2D array of k-values.
        """
        return Kmatrix(self.N, self.scale).kM

    def _sigMatrix(self, As, ns, kp):
        psM = ScaleInvariantPSM(self.kM, As, ns, kp).psM
        return np.sqrt(psM)/np.sqrt(2)   # Dividing by sqrt(2) to account for the fft field being complex valued

    def _complexGaussMatrix(self, mu=0, sig=1):
        x = np.random.normal(mu, sig, (self.N, self.N//2 + 1))
        y = np.random.normal(mu, sig, (self.N, self.N//2 + 1))
        return x + (y * 1j)

    def _fftMatrix(self):
        gaussM = self._complexGaussMatrix()
        return self._enforceRealSymmetries(gaussM * self._sigM)

    def _enforceRealSymmetries(self, fftM):

        # Setting divergent point to 0 (this point represents mean of real field so this is reasonable)
        fftM[0, 0] = 0

        # Ensuring Nyquist points are real
        fftM[self.N//2, 0] = np.real(fftM[self.N//2, 0]) * np.sqrt(2)
        fftM[0, self.N//2] = np.real(fftM[0, self.N//2]) * np.sqrt(2)
        fftM[self.N//2, self.N//2] = np.real(fftM[self.N//2, self.N//2]) * np.sqrt(2)

        # +ve k_y mirrors -ve k_y at k_x = 0
        fftM[self.N//2 + 1:, 0] = np.conjugate(fftM[1:self.N//2, 0][::-1])

        # +ve k_y mirrors -ve k_y at k_x = N/2 (Nyquist freq) !!!!!!!!!!! Why?
        fftM[self.N//2 + 1:, -1] = np.conjugate(fftM[1:self.N//2, -1][::-1])

        return fftM
