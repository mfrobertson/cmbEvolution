import camb
import numpy as np
from camb.symbolic import *


class TransferFuncs:

    def __init__(self, ks, conformalTimes, source="monopole"):

        self.ks = ks
        self.times = conformalTimes
        self.source = source
        self.tfs = self.generateFuncs()

    def generateFuncs(self):

        pars = camb.CAMBparams()
        #pars.set_accuracy(AccuracyBoost=2)
        pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122)
        data = camb.get_transfer_functions(pars)

        if self.source == "monopole":
            monopole_source = get_scalar_temperature_sources()[0]
            transferFuncs = data.get_time_evolution(self.ks, self.times, [monopole_source/visibility])
            return transferFuncs

    def saveFuncs(self, name):
        np.save(f"TFs/{name}", self.tfs)
