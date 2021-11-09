import numpy as np

if __name__ == "__main__":

    # Create video
    from videobuilder import Video
    N = 1024                         # NxN field
    scale = 1000                     # Width of field [Mpc]
    ks = np.logspace(-4, 1, 1000)    # Sample wavenumbers for transfer function [Mpc^-1]
    etas = range(1, 281)             # Sample conformal times for transfer functions [Mpc]
    clim = 3e-7                      # Can set max and min color range to stop scale varying throughout video
    loc = r"C:path\to\directory"     # Output directory

    Video(f"monopole_video_{N}_{scale}", loc=loc, N=N, scale=scale, ks=ks, etas=etas, delPics=True, clims=(clim, -clim))
