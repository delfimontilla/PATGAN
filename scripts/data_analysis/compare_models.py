import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

LABELS = ("Datos degradados", "RealESRGAN M1- patches", "RealESRGAN M1- full sinogram",
           "RealESRGAN M2 - patches", "RealESRGAN M2 - full sinogram", 
           "RealESRGAN M3 - patches","RealESRGAN M3 - full sinogram", "U-Net", "Datos originales")
COLORS = ("peru", "palegoldenrod", "khaki", 
            "pink", "lightpink",
            "paleturquoise", "powderblue", 
            "lavender", "lightgreen")

def plot_bar_error(data, yerr, ylabel = None, title = None, filename = None, bestcase = False, noisy = False):
    """Saves a plot with colorbar for a given data array.

    Args:
        data (ndarray): Data array to plot.
        maximum (float, optional): maximum value to plot. Defaults to None.
        path (str, optional): saving directory. Defaults to None.
        filename (str, optional): filename to use to use plot. Defaults to None.
    """
    fig, ax = plt.subplots(constrained_layout = True)
    #fig.set_size_inches(20,14)
    N = np.arange(len(data))
    if bestcase:
        colors = COLORS[:-1]
        labels = LABELS[:-1]
    else:
        colors = COLORS
        labels = LABELS
    plt.bar(N, data, yerr=yerr, color=colors, alpha=1, align='center', ecolor='slategray', capsize=5)
    plt.xticks(N, labels, rotation=80, fontsize = 8)
    if title:
        plt.title(title)
    if ylabel:
        plt.ylabel(ylabel)
    if yerr is not None:
        min = np.amin(data) - 3*np.amax(yerr)
        max = np.amax(data) + 3*np.amax(yerr)
    else:
        min = np.amin(data) - 0.15*np.amax(data)
        max = np.amax(data) + 0.15*np.amax(data)
    plt.ylim(min, max)
    if filename:
        if noisy:
            filename = f"{filename}_60db"
        if bestcase: 
            filename = f"{filename}_bestcase"
        plt.savefig(f"{filename}.png")
    #plt.show()
    plt.close()


if __name__ == '__main__':
    #TYPE = "Dataset de testeo"
    TYPE = "Patron"
    
    fname = "imagestable.csv"
    df = pd.read_csv(fname, header=2, index_col=0)#, usecols=[0,1,2,3,4,5,6,7,8])
    
    TESTING_IMAGES = ["Vasos sanguíneos", "Tejido mamario", "Derenzo", "PAT", "TOA"]
    label ={"PC": "Correlación de Pearson", 
                "SSIM": "Índice de similitud estructural", 
                "RMSE": "Raíz del error cuadrático medio", 
                "PSNR": "Relación señal a ruido pico", 
            "PC.1": "Correlación de Pearson", 
                "SSIM.1": "Índice de similitud estructural", 
                "RMSE.1": "Raíz del error cuadrático medio", 
                "PSNR.1": "Relación señal a ruido pico",
            }
    
    if TYPE == "Dataset de testeo":
        df = pd.read_csv(fname, header=2, index_col=0, usecols=[0,1,2,3,4,5,6,7,8])
        col_name = ["PC","std","SSIM","std.1","RMSE","std.2","PSNR","std.3"]
    else:
        df = pd.read_csv(fname, header=2, index_col=0)
        col_name = ["PC", "SSIM", "RMSE", "PSNR", "PC.1", "SSIM.1", "RMSE.1", "PSNR.1"]
        noisy_cols = ["PC.1", "SSIM.1", "RMSE.1", "PSNR.1"]

    if TYPE == "Dataset de testeo":
        bestcase = True
        for metric, std in zip(col_name[::2], col_name[1::2]):
            data = np.array(df[metric].values [:8]).astype(np.float32)
            yerr = np.array(df[std].values[:8]).astype(np.float32)
            plot_bar_error(data, yerr, ylabel = f"{label[metric]}", title = f"{TYPE}", filename = f"{TYPE} - {label[metric]}", bestcase = bestcase)
        bestcase = False
        for metric, std in zip(col_name[::2], col_name[1::2]):
            data = np.array(df[metric].values [12:]).astype(np.float32)
            yerr = np.array(df[std].values[12:]).astype(np.float32)
            plot_bar_error(data, yerr, ylabel = f"{label[metric]}", title = f"{TYPE}", filename = f"{TYPE} - {label[metric]}", bestcase = bestcase)
    
    else:
        bestcase = True
        init = 0
        step = 8
        for img_type in TESTING_IMAGES:
            for metric in col_name:
                data = np.array(df[metric].values [init:init+step]).astype(np.float32)
                noisy = True if metric in noisy_cols else False
                plot_bar_error(data, None, ylabel = f"{label[metric]}", title = f"{img_type}", filename = f"{img_type} - {label[metric]}", bestcase = bestcase, noisy=noisy)
            init = init + step + 1
        bestcase = False
        init = init + 3
        step = 9
        for img_type in TESTING_IMAGES:
            for metric in col_name:
                data = np.array(df[metric].values [init:init+step]).astype(np.float32)
                noisy = True if metric in noisy_cols else False
                plot_bar_error(data, None, ylabel = f"{label[metric]}", title = f"{img_type}", filename = f"{img_type} - {label[metric]}", bestcase = bestcase, noisy=noisy)
            init = init + step + 1
