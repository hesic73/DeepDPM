#
# Created on March 2022
#
# Copyright (c) 2022 Meitar Ronen
#

from matplotlib import pyplot as plt
import numpy as np

class PlotUtils:
    @staticmethod
    def plot_weights_histograms(K, pi, start_sub_clustering, current_epoch, pi_sub, for_thesis=False):
        fig = plt.figure(figsize=(10, 3))
        ind = np.arange(K)
        plt.bar(ind, pi, label="clusters' weights", align="center", alpha=0.3)
        if start_sub_clustering <= current_epoch and pi_sub is not None:
            pi_sub_1 = pi_sub[0::2]
            pi_sub_2 = pi_sub[1::2]
            plt.bar(ind, pi_sub_1, align="center", label="sub cluster 1")
            plt.bar(
                ind, pi_sub_2, align="center", bottom=pi_sub_1, label="sub cluster 2"
            )

        plt.xlabel("Clusters inds")
        plt.ylabel("Normalized weights")
        plt.title(f"Epoch {current_epoch}: Clusters weights")
        if for_thesis:
            ax = plt.gca()
            ax.axes.xaxis.set_ticks([])
            ax.axes.yaxis.set_ticks([])
        else:
            plt.legend()
        return fig

    