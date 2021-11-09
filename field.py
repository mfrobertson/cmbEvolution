import numpy as np
from scipy import fft
import matplotlib.pyplot as plt


class Field:

    def __init__(self, field, N, scale, space):
        self.field = field
        self.N = N
        self.scale = scale
        self.space = space

    def FFT(self):
        fieldFFT = fft.rfftn(self.field)
        return Field(fieldFFT, self.N, self.scale, "fourier")

    def iFFT(self):
        field = fft.irfftn(self.field)
        return Field(field, self.N, self.scale, "physical")

    def kMatrix(self):
        return Kmatrix(self.N, self.scale).kM

    def drawField(self, title=None, clims=None, cbar=True):
        if self.space == "physical":
            extent = [0, self.scale, 0, self.scale]
            field = self.field
            xlabel = "$x$ [Mpc]"
            ylabel = "$y$ [Mpc]"
        else:
            extent = [0, np.pi * self.N / self.scale, np.pi * self.N / self.scale, 0]
            field = np.abs(self.field[:self.N // 2, :])
            xlabel = "$k_x$ [Mpc$^{-1}$]"
            ylabel = "$k_y$ [Mpc$^{-1}$]"

        plt.figure()
        plt.imshow(field, extent=extent, cmap="jet")
        cb = plt.colorbar()

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

    def __init__(self, N, scale):
        self.N = N
        self.scale = scale
        self.kM = self.kMatrix()

    def kMatrix(self):
        kx = fft.rfftfreq(self.N, d=self.scale / self.N) * 2 * np.pi
        ky = fft.fftfreq(self.N, d=self.scale / self.N) * 2 * np.pi
        kSqr = kx[np.newaxis, ...] ** 2 + ky[..., np.newaxis] ** 2
        return np.sqrt(kSqr)
