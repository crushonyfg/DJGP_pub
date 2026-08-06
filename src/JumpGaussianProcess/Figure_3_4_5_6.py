from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from skimage import measure
from datetime import datetime
import os
import argparse

from simulate_case import simulate_case
from JumpGP_LD import JumpGP_LD

@dataclass
class FigureParams:
    percent_train: float = 0.5
    sig: float = 2
    caseno: int = 1
    sc: float = 0.025 / 0.02
    n_neighbors: int = 40
    path: str = 'D:/new_windows/PhD/spring2025/park/JumpGP_code_py/results'

# ... keep check_and_reshape function ...

def generate_figures(params: FigureParams):
    if not os.path.exists(params.path):
        os.makedirs(params.path)
    # simulate case
    x, y, xt, yt, y0, gx, r, bw = simulate_case(params.caseno, params.sig, params.percent_train)

    L = len(gx)
    
    my = np.mean(yt)
    y -= my
    yt -= my
    y0 -= my

    bw = np.reshape(bw, (L, L))

    # define test points for different cases
    xs_cases = {
        1: np.array([[20, 33], [16, 16], [7, 5], [25, 36], [8, 32], [38, 18]]),
        2: np.array([[7, 22], [17, 22], [35, 22], [5, 28], [22, 35], [31, 7]]),
        3: np.array([[37, 11], [17, 24], [21, 21], [36, 36], [8, 35], [34, 32]]),
        4: np.array([[37, 11], [36, 36], [8, 35], [17, 24], [21, 21], [34, 32]])
    }
    xs = xs_cases[params.caseno]

    # normalize test points
    xs = xs / len(np.arange(0, 1.025, 0.025)) - 0.5

    Nt = xs.shape[0]
    subplot_layout = (2, 4)
    sel = [0, 1, 2, 5]  # Index selection for test points
    loc = list(range(1, 9))

    current_time = datetime.now().strftime("%Y%m%d_%H")

    for j in range(4):
        k = 1 if j < 4 else 1
        xt = xs[j, :]  # test point for each subplot

        # find nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=params.n_neighbors, algorithm='auto').fit(x)
        idx = nbrs.kneighbors([xt], return_distance=False)[0]
        lx, ly = x[idx, :], y[idx]
        ly = ly.reshape(-1, 1)
        xt = xt.reshape(1, -1)

        if k == 0:
            mu_t, sig2_t, model, h3 = JumpGP_LD(lx, ly, xt, 'CEM', True)
        else:
            mu_t, sig2_t, model, h3 = JumpGP_LD(lx, ly, xt, 'SEM', True)

        plt.imshow(np.reshape(yt, (L, L)), cmap='gray', extent=(gx[0], gx[-1], gx[-1], gx[0]))
        
        # plot local training points
        plt.scatter(lx[:, 0], lx[:, 1], color='r', marker='+', s=100, label='Local Training Inputs')
        
        # plot test point
        plt.scatter(xt[0,0], xt[0,1], color='c', marker='o', s=100, label='Test Point')
        
        current_ax = plt.gca()
        
        # function for copying PathCollection
        def copy_path_collection(artist, ax):
            # get the properties of PathCollection
            offsets = artist.get_offsets()  # get the coordinates of the points
            sizes = artist.get_sizes()  # get the size of the points
            facecolors = artist.get_facecolor()  # get the color of the points
            edgecolors = artist.get_edgecolor()  # get the edge color
        
            # plot the scatter plot again
            return ax.scatter(offsets[:, 0], offsets[:, 1], s=sizes, c=facecolors, 
                            edgecolor=edgecolors, alpha=artist.get_alpha(), label=artist.get_label())
        
        # plot h3 again
        new_artist = copy_path_collection(h3[0], current_ax)
        a = np.array([[1, -0.5], [1, 0.5]])
        b_plot = -np.dot(a, model['w'][0:2]) / model['w'][2]
        # current_ax.plot(a, b_plot, 'r', linewidth=2)
        current_ax.plot(np.array([-0.5,0.5]), b_plot, 'r', linewidth=3)
        # line = h3[1][1]
        # current_ax.plot(line.get_xdata(), line.get_ydata(),
        #                 color=line.get_color(), linewidth=line.get_linewidth(),
        #                 label=line.get_label())
        plt.imshow(np.reshape(yt, (L, L)), cmap='gray', extent=(gx[0], gx[-1], gx[-1], gx[0]))
        # add legend
        plt.legend()

        plt.savefig(f'{params.path}caseno_{params.caseno}_figure_{j}_{current_time}.png', dpi=300, bbox_inches='tight')
        plt.clf()

# example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='generate figures')
    parser.add_argument('--percent_train', type=float, default=0.5, help='percentage of training data')
    parser.add_argument('--sig', type=float, default=2, help='signal strength')
    parser.add_argument('--caseno', type=int, default=2, help='case number')
    parser.add_argument('--sc', type=float, default=0.025/0.02, help='scaling factor')
    parser.add_argument('--n_neighbors', type=int, default=40, help='number of neighbors')
    parser.add_argument('--path', type=str, default='D:/new_windows/PhD/spring2025/park/JumpGP_code_py/results', help='path to save results')

    args = parser.parse_args()

    params = FigureParams(
        percent_train=args.percent_train,
        sig=args.sig,
        caseno=args.caseno,
        sc=args.sc,
        n_neighbors=args.n_neighbors,
        path=args.path
    )
    generate_figures(params)
