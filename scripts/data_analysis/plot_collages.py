import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib_scalebar.scalebar import ScaleBar
from scipy.io import loadmat
from pydicom import dcmread
from PIL import Image
import numpy as np


def imfrommat(img_bytes):
    f = loadmat(img_bytes)
    if "sino" in f:
        sino = f["sino"]
    elif "pred" in f:
        sino = f["pred"]
    elif "full_sino" in f:
        sino = f["full_sino"]
    elif "img" in f:
        sino = f["img"]
    elif "img_patch" in f:
        sino = f["img_patch"]
    elif "unet" in f:
        sino = f["unet"]
    elif "realesrgan" in f:
        sino = f["realesrgan"]
    else:
        raise Exception("Error opening mat file, incorrect field name") 
    sino = np.array([sino])[0]
    return sino


name = "PAT"
images_path_dir = Path(f"images/{name}")


if name == "test_dataset":
    titles_first_row3 = ["Referencia", "Sin pre-procesamiento", "U-Net - P"]
    titles_first_row4 = ["Original"]
    titles_first_row4 = titles_first_row4 + titles_first_row3
    titles_gan = ["GAN M1 - C", "GAN M2 - C", "GAN M3 - C", "GAN M1 - P", "GAN M2 - P", "GAN M3 - P",]

    images_gan = images_path_dir.glob("M*")
    images_gan = list(images_gan)
    images_gan.sort()
    images_gan = images_gan[::2] + images_gan[1::2]
    images_gan = [imfrommat(str(image)) for image in images_gan]

    first_row_3 = []
    first_row_3.append(list(images_path_dir.glob("GT*"))[0])
    first_row_3.append(list(images_path_dir.glob("LQ*"))[0])
    first_row_3.append(list(images_path_dir.glob("UNET*"))[0])
    first_row_3 = [imfrommat(str(image)) for image in first_row_3]

    first_row_4 = []
    file = str(list(images_path_dir.glob(f"{name}*"))[0])
    orig_img = dcmread(file)
    orig_img = orig_img.pixel_array
    #Downsize image to fit the system's visual field
    orig_img = np.array(Image.fromarray(orig_img).resize((401,401)))
    #Normalize min to 0 and max a 1
    img = (orig_img-np.min(orig_img))/(np.max(orig_img)-np.min(orig_img))
    first_row_4.append(img)
    first_row_4 = first_row_4 + first_row_3
    #GAN plot
    rows = 2
    cols = 3
    fig, axs = plt.subplots(nrows = rows, ncols = cols, tight_layout = True, figsize=(14,10))
    for title, image, ax in zip(titles_gan, images_gan, axs.ravel()):
        ax.imshow(image, "gray")
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
        ax.tick_params(bottom=False, left=False, top=False, right=False)
        ax.set_title(title, fontweight="bold", size=22)
    plt.subplots_adjust(hspace=0.05, wspace=0.05)
    plt.savefig(f'{name}_gan', bbox_inches='tight')    
    #plt.show()


    #first row 3
    rows = 1
    cols = 3
    fig, axs = plt.subplots(nrows = rows, ncols = cols, tight_layout = True, figsize=(14,8))
    for title, image, ax in zip(titles_first_row3, first_row_4[1:], axs.ravel()):
        ax.imshow(image, "gray")
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
        ax.tick_params(bottom=False, left=False, top=False, right=False)
        ax.set_title(title, fontweight="bold", size=16)
    plt.subplots_adjust(hspace=0.05, wspace=0.05)
    plt.savefig(f'{name}_first3', bbox_inches='tight')    
    #plt.show()

    #first row 3
    rows = 1
    cols = 4
    fig, axs = plt.subplots(nrows = rows, ncols = cols, tight_layout = True, figsize=(14,8))
    for title, image, ax in zip(titles_first_row4, first_row_4, axs.ravel()):
        ax.imshow(image, "gray")
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
        ax.tick_params(bottom=False, left=False, top=False, right=False)
        ax.set_title(title, fontweight="bold", size=16)
    plt.subplots_adjust(hspace=0.05, wspace=0.05)
    plt.savefig(f'{name}_first4', bbox_inches='tight')    
    #plt.show()

else:
    titles_gan = ["Sin pre-procesamiento", "GAN M1 - C", "GAN M2 - C", "GAN M3 - C", 
                  "U-Net - P", "GAN M1 - P", "GAN M2 - P", "GAN M3 - P",]

    images_gan = images_path_dir.glob("M*")
    images_gan = list(images_gan)
    images_gan.sort()
    sin_proc = list(images_path_dir.glob("LQ*"))
    unet = list(images_path_dir.glob("UNET*"))
    images = sin_proc + images_gan[::2] + unet + images_gan[1::2]
    images = [imfrommat(str(image)) for image in images]

    rows = 2
    cols = 4
    fig, axs = plt.subplots(nrows = rows, ncols = cols, tight_layout = True, figsize=(18,10))
    for title, image, ax in zip(titles_gan, images, axs.ravel()):
        ax.imshow(image, "gray")
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
        ax.tick_params(bottom=False, left=False, top=False, right=False)
        ax.set_title(title, fontweight="bold", size=22)
    plt.subplots_adjust(hspace=0.05, wspace=0.05)
    plt.savefig(f'{name}_collage', bbox_inches='tight')    

    if name == "PAT":
        titles_first = ["Original", "Referencia"]
        gt_recon = list(images_path_dir.glob("GT*"))[0]
        first_row_1 = imfrommat(str(gt_recon))

        file = str(list(images_path_dir.glob(f"{name}*"))[0])
        image = Image.open(file)
        image = image.convert("L")
        orig_img = image.resize((201,201))
        img = np.array(orig_img)
        img = img.astype(np.float64)
        orig = img/np.amax(img)
        img = 1 - orig
        first_row = [img, first_row_1]

        scale = 20.1/201
        rows = 1
        cols = 2
        fig, axs = plt.subplots(nrows = rows, ncols = cols, tight_layout = True, figsize=(15,7))
        i = 0
        for title, image, ax in zip(titles_first, first_row, axs.ravel()):
            ax.imshow(image, "gray")
            ax.axes.xaxis.set_ticklabels([])
            ax.axes.yaxis.set_ticklabels([])
            ax.tick_params(bottom=False, left=False, top=False, right=False)
            ax.set_title(title, fontweight="bold", size=16)
            if i ==0:
                bar = ScaleBar(scale, 'mm', frameon=False, color='white', 
                               fixed_value=5, location='lower left',
                               font_properties = {"size":16})
                ax.add_artist(bar)
            i += 1
        plt.subplots_adjust(hspace=0.05, wspace=0.05)
        plt.savefig(f'{name}_scale', bbox_inches='tight')   
