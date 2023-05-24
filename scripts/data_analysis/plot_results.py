import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def open_csv(filename):
    """Open and load csv file.
    Args:
        filename (str or PATH): path to csv file.
    Returns:
        df (DataFrame): pandas DataFrame.
    """
    df = pd.read_csv(filename).dropna()
    return df


def plot_scatter(axis, intensity = None, title = None, 
                figure_size = (12, 8), show_plot = False, psnr_plot = False):
    """Plot in grayscale image represented by numpy array.
    Args:
        img (ndarray): 2D or 3D array.
        intensity (ndarray): 1D array. Defaults to None.
        title (string, optional): plot title. Defaults to None.
        figure_size (tuple, optional): width and height of resulting figure. 
            Defaults to (10,10).
        show_plot (bool): Whether to show plot or not. Default to False.
        psnr_plot (bool): Whether it is plotting PSNR or not. Default to False.
    """
    plt.figure(figsize=figure_size)
    plt.scatter(axis, intensity, c = intensity)
    if np.amax(intensity) == np.inf:
        filtered_intensity = filter(lambda i: i != np.inf, intensity)
        mean_int = np.mean(list(filtered_intensity))
        filtered_intensity = filter(lambda i: i != np.inf, intensity)
        std_int = np.std(list(filtered_intensity))
    else:
        mean_int = np.mean(intensity)
        std_int = np.std(intensity)
    print(mean_int)
    print(std_int)
    plt.axhline(y=mean_int, color = "b", linestyle = "-.", label="Promedio")
    plt.fill_between(axis, mean_int-std_int, mean_int+std_int, alpha=0.2)
    plt.legend()
    plt.xlim(0,len(intensity))
    if psnr_plot:
        ax = plt.gca()
        labels = [item.get_text() for item in ax.get_yticklabels()]
        labels[-1] = 'inf'
        ax.set_yticklabels(labels)
    if title is not None:
        words = title.split()
        save_filename = "_".join(words)
        plt.title(title)
        plt.grid(visible = True)
        plt.tight_layout()
        plt.savefig(save_filename+"images.png", format="PNG")
    if show_plot:
        plt.show()
    plt.close()


if __name__ == '__main__':
    fname = "PAT_GAN/results/res_pretrained_recon/results_IMAGES.csv"
    res = open_csv(fname)
    axis = range(len(res.index))
    plot_scatter(axis, res.PC, title = "Correlacion de Pearson", show_plot=False)
    plot_scatter(axis, res.ssmval, title = "Indice de similitud estructural", show_plot=False)
    plot_scatter(axis, res.rmse, title = "Raíz del error cuadrático medio", show_plot=False)
    psnr = res.peaksnr.copy()
    for x in psnr:
        if x == np.inf:
            x = 200
    plot_scatter(axis, psnr, title = "Relación señal-ruido pico", show_plot=False, psnr_plot=True)
