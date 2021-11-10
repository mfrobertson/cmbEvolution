import numpy as np
import matplotlib.pyplot as plt
from realisation import FieldRealisation
from transfers import TransferFuncs
import os
import platform

class Video:

    def __init__(self, name, loc, N=1024, scale=1000, ks=np.logspace(-4, 1, 1000), etas=range(1,300), delPics=True, clims=None):
        self.name = name
        self.loc = loc
        self.N = N
        self.scale = scale
        self.ks = ks
        self.etas = etas
        self.clims = clims
        self.fileSep = self.getFileSep()

        picDir = self.buildVideo()

        if delPics:
            self.delDir(picDir)

    def getFileSep(self):
        if platform.system() == "Windows":
            return r"\\"
        else:
            return r"/"

    def buildVideo(self):
        fr = FieldRealisation(self.N, self.scale)
        fr.buildSI()

        picDir = self.drawPics(fr)

        os.chdir(os.getcwd())
        os.chdir(self.loc)
        files = picDir + self.fileSep + "field_%02d.png"
        os.system(fr'ffmpeg -i {files} -filter:v "setpts=2*PTS" -r 30 -pix_fmt yuv420p -y {self.name}.mp4')

        return picDir

    def drawPics(self, fr):
        dir = fr"{self.loc + self.fileSep}pics_{self.N}_{self.scale}"
        if not os.path.isdir(self.loc):
            os.mkdir(self.loc)
            os.mkdir(dir)
        elif not os.path.isdir(dir):
            os.mkdir(dir)

        tf_raw = TransferFuncs(self.ks, self.etas).tfs[:, :, 0]
        for iii, eta in enumerate(self.etas):

            fr.calcField(eta, tf_raw=tf_raw[:,iii])
            fr.realField.drawField(clims=self.clims, cbar=False)

            plt.gca().set_axis_off()
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.savefig(rf"{dir + self.fileSep}field_%02d.png" % eta, bbox_inches='tight', pad_inches=0)

            plt.close()
        return dir

    def delDir(self, dir):
        os.chdir(os.getcwd())
        os.chdir(self.loc)
        os.system(f"rd /Q /S {dir}")
