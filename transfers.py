import camb
from camb.symbolic import *


class TransferFuncs:
    """
    Calculates CAMB transfer function.

    Attributes
    ----------
    ks : list or ndarray
        Sample k-values at which transfer function is calculated at.
    etas : list or ndarray
        Sample conformal times at which transfer function is calculated at.
    source : str
        Name of CAMB transfer function to calculate.
    tfs : ndarray
        3D array of calculated transfer functions, [ks, etas, source].
    """

    def __init__(self, ks, etas, source="monopole"):
        """
        Constructor.

        Parameters
        ----------
        ks : list or ndarray
            Sample k-values at which transfer function is calculated at.
        etas : list or ndarray
            Sample conformal times at which transfer function is calculated at.
        source : str
            Name of CAMB transfer function to calculate.
        """

        self.ks = ks
        self.etas = etas
        self.source = source
        self.tfs = self._generateFuncs()

    def _generateFuncs(self):
        pars = camb.CAMBparams()
        #pars.set_accuracy(AccuracyBoost=2)
        pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122)
        data = camb.get_transfer_functions(pars)

        if self.source == "monopole":
            monopole_source = get_scalar_temperature_sources()[0]
            transferFuncs = data.get_time_evolution(self.ks, self.etas, [monopole_source/visibility])
            return transferFuncs

        transferFuncs = data.get_time_evolution(self.ks, self.etas, [self.source])
        return transferFuncs