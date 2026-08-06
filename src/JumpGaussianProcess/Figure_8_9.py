from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
from sklearn.neighbors import NearestNeighbors
from scipy.ndimage import label
from datetime import datetime
import os

from simulate_case import simulate_case
from JumpGP_QD import JumpGP_QD
from JumpGP_LD import JumpGP_LD
from calculate_gx import calculate_gx

@dataclass
class FigureParams:
    percent_train: float = 0.5
    sig: float = 2
    caseno: int = 6  # use 4 for figure 8, use 6 for figure 9
    n_neighbors: int = 40
    path: str = 'D:/new_windows/PhD/spring2025/park/JumpGP_code_py/results'

def generate_figure(params):
    if not os.path.exists(params.path):
        os.makedirs(params.path)
    # generate data
    x, y, xt, yt, y0, gx, r, bw = simulate_case(params.caseno, params.sig, params.percent_train)
    L = len(gx)

    # center the data
    my = np.mean(yt)
    y, yt, y0 = y - my, yt - my, y0 - my

    # get the boundary
    bw = bw.reshape(L, L)

    # define test points
    xs_cases = {
        4: np.array([[37, 11], [36, 36], [8, 35], [17, 24], [21, 21], [34, 32]]),
        6: np.array([[27, 21], [10, 27], [37, 31], [9, 19], [21, 22], [27, 15]])
    }
    xs = xs_cases[params.caseno] / len(np.arange(0, 1.025, 0.025)) - 0.5

    # plot settings
    # fig, axs = plt.subplots(3, 4, figsize=(15, 15))
    sel = [1, 2, 3, 6]

    for j in range(4):
        # ax = axs[(j - 1) // 4, (j - 1) % 4]
        
        k = 1 if j < 4 else 1
        xt = xs[j, :]

        # find k nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=params.n_neighbors, algorithm='auto').fit(x)
        idx = nbrs.kneighbors([xt], return_distance=False)[0]
        lx, ly = x[idx, :], y[idx]
        ly = ly.reshape(-1, 1)
        xt = xt.reshape(1, -1)

        # select JumpGP function
        if k == 0:
            mu_t, sig2_t, model, h3 = JumpGP_QD(lx, ly, xt, 'CEM', True)
        else:
            mu_t, sig2_t, model, h3 = JumpGP_QD(lx, ly, xt, 'VEM', True)

        plt.imshow(np.reshape(yt, (L, L)), cmap='gray', extent=(gx[0], gx[-1], gx[-1], gx[0]))
        plt.scatter(lx[:, 0], lx[:, 1], color='r', marker='+', s=100, label='Local Training Inputs')
        
        plt.scatter(xt[0, 0], xt[0, 1], color='c', marker='o', s=100, label='test point')
        current_ax = plt.gca()
        
        # function to copy PathCollection
        def copy_path_collection(artist, ax):
            if isinstance(artist, list):
                # if artist is a list, process each element separately
                new_artists = []
                for art in artist:
                    if isinstance(art, PathCollection):  # use the correct PathCollection class
                        offsets = art.get_offsets()
                        sizes = art.get_sizes()
                        facecolors = art.get_facecolor()
                        edgecolors = art.get_edgecolor()
                        new_art = ax.scatter(offsets[:, 0], offsets[:, 1], s=sizes, c=facecolors, 
                                          edgecolor=edgecolors, alpha=art.get_alpha(), label=art.get_label())
                        new_artists.append(new_art)
                    else:  # for other types of artist (including contour), use directly
                        new_artists.append(art)
                return new_artists
            else:
                # if artist is a single PathCollection
                if isinstance(artist, PathCollection):
                    offsets = artist.get_offsets()
                    sizes = artist.get_sizes()
                    facecolors = artist.get_facecolor()
                    edgecolors = artist.get_edgecolor()
                    return ax.scatter(offsets[:, 0], offsets[:, 1], s=sizes, c=facecolors, 
                                   edgecolor=edgecolors, alpha=artist.get_alpha(), label=artist.get_label())
                else:
                    return artist
        
        # redraw h3
        new_artist = copy_path_collection(h3, current_ax)

        d = x.shape[1]
        g1, g2 = np.meshgrid(np.arange(d), np.arange(d))
        g1, g2 = g1.flatten(order='F'), g2.flatten(order='F')
        id_mask = g1 >= g2
        gx_range = np.linspace(0, 1, 400) - 0.5
        ptx, pty = np.meshgrid(gx_range, gx_range)
        allx = np.hstack([ptx.ravel().reshape(-1, 1), pty.ravel().reshape(-1, 1)])
        allx = np.hstack([allx, (allx[:, g1[id_mask]] * allx[:, g2[id_mask]])])
        gx0, _ = calculate_gx(allx, model['w'])
        gy = np.sign(gx0)
        # Visualization logic can go here, for example:
        # print(f"Plotting results for mode: {mode}")
        L1 = len(gx_range)
        gy_reshaped = np.reshape(gy, (L1, L1))

        # plot contour
        plt.contour(gx_range, gx_range, gy_reshaped, levels=[0], colors='r')

        # if i == 1 and k == 1:
        #     ax.set_xlabel('x1')
        #     ax.set_ylabel('x2')

        plt.legend()
        # plt.show()
        current_time = datetime.now().strftime("%Y%m%d_%H")
        plt.savefig(f'{params.path}caseno_{params.caseno}_figure_{j}_{current_time}.png', dpi=300, bbox_inches='tight')
        plt.clf()

if __name__ == "__main__":
    params = FigureParams()
    generate_figure(params)
