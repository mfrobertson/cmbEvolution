import matplotlib.pyplot as plt
from realisation import FieldRealisation
from transfers import TransferFuncs
import os
import platform

class Video:
    """
    Generates video of monopole source evolution, and saves output as mp3 file.

    Attributes
    ----------
    name : str
        Output filename.
    loc : str
        Output directory.
    N : int
        Number of pixels of array.
    scale : int or float
        Physical length of field [Mpc].
    ks : array
        Sample k-values at which transfer function is calculated at.
    etas : array
        Sample conformal times at which transfer function is calculated at.
    clims : 2-tuple
        Colorbar limits.
    """

    def __init__(self, name, loc, N=1024, scale=1000, ks=None, etas=range(1,300), delPics=True, clims=None):
        """
        Constructor.

        Parameters
        ----------
        name : str
            Output filename.
        loc : str
            Output directory.
        N : int
            Number of pixels of array.
        scale : int or float
            Physical length of field [Mpc].
        ks : array
            Sample k-values at which transfer function is calculated at.
        etas : array
            Sample conformal times at which transfer function is calculated at.
        delPics : bool
            Delete directory containing png snapshots.
        clims : 2-tuple
            Colorbar limits.
        """
        self.name = name
        self.loc = loc
        self.N = N
        self.scale = scale
        self.ks = ks
        self.etas = etas
        self.clims = clims
        self._fileSep = self._getFileSep()

        picDir = self._buildVideo()

        if delPics:
            self._delDir(picDir)

    def _getFileSep(self):
        if platform.system() == "Windows":
            return r"\\"
        else:
            return r"/"

    def _buildVideo(self):
        fr = FieldRealisation(self.N, self.scale)
        fr.buildSI()

        if self.ks is None:
            self.ks = fr.siField.sampleKs()

        picDir = self._drawPics(fr)

        os.chdir(os.getcwd())
        os.chdir(self.loc)
        files = picDir + self._fileSep + "field_%02d.png"
        os.system(fr'ffmpeg -i {files} -filter:v "setpts=2*PTS" -r 30 -pix_fmt yuv420p -y {self.name}.mp4')

        return picDir

    def _drawPics(self, fr):
        dir = fr"{self.loc + self._fileSep}pics_{self.N}_{self.scale}"
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
            plt.savefig(rf"{dir + self._fileSep}field_%02d.png" % eta, bbox_inches='tight', pad_inches=0)

            plt.close()
        return dir

    def _delDir(self, dir):
        os.chdir(os.getcwd())
        os.chdir(self.loc)
        os.system(f"rd /Q /S {dir}")
