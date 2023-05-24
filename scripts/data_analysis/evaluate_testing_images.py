import numpy as np
from pathlib import Path
from scipy.io import loadmat
import csv
import matplotlib.pyplot as plt 
from natsort import natsorted
from PIL import Image

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy.stats import pearsonr


def imfrommat(img_bytes):
    f = loadmat(img_bytes)
    if "sino" in f:
        sino = f["sino"]
    elif "limited_noise_interpolated" in f:
        sino = f["limited_noise_interpolated"]
    elif "img_patch" in f:
        sino = f["img_patch"]
    elif "unet" in f:
        sino = f["unet"]
    elif "realesrgan" in f:
        sino = f["realesrgan"]
    elif "pred" in f:
        sino = f["pred"]
    elif "img" in f:
        sino = f["img"]
    else:
        raise Exception("Error opening mat file, incorrect field name") 
    sino = np.array([sino])
    return sino

bestcase = False
sino_comp = False

if bestcase:
    gt_dir = "/PAT_GAN/datasets/testing_images/sino_gt_recon/"
elif sino_comp:
    gt_dir = "/PAT_GAN/datasets/testing_images/sino_gt/"
else:
    gt_dir = "/PAT_GAN/datasets/testing_images/images_gt/"
load_path = Path(gt_dir)

if bestcase or sino_comp:
    gt_files = list(load_path.glob("*.mat"))
else:
    gt_files = list(load_path.iterdir())
gt_files = natsorted(gt_files)

recon_dir = "/PAT_GAN/datasets/testing_images/sino_realesrgan_recon/"
recon_path = Path(recon_dir)
recon_files = list(recon_path.glob("*.mat"))
recon_files = natsorted(recon_files)

results = []

#Evaluation of testing images dataset
for recon_file, gt_file in zip(recon_files, gt_files):
    filename = Path(recon_file).stem
    recon = imfrommat(recon_file)
    norm_recon = recon[0].astype(np.float64)
    
    if not sino_comp:
        #normalize reconstructions
        norm_recon = norm_recon/np.amax(norm_recon)

    print(gt_file)
    print(recon_file)
    if bestcase or sino_comp:
        image = imfrommat(gt_file)
        image = image[0].astype(np.float64)
        orig = image/np.amax(image)
    else:
        image = Image.open(gt_file)
        image = image.convert("L")
        orig_img = image.resize((201,201))
        img = np.array(orig_img)
        img = img.astype(np.float64)
        #normalize original images
        orig = img/np.amax(img)
        # change background to 0 and object to 1
        if "inv" not in filename:
            orig = 1 - orig
    

    SSIM = ssim(orig, norm_recon)
    RMSE = np.sqrt(mse(orig, norm_recon))
    PSNR = psnr(orig, norm_recon, data_range=1)
    PC = pearsonr(orig.flatten(), norm_recon.flatten())
    result = {"file": filename, "PC": PC[0], "ssmval": SSIM, "rmse": RMSE, "peaksnr": PSNR}
    results.append(result)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,7))
    ax1.imshow(orig, "gray")
    ax1.axis('off')
    ax2.imshow(norm_recon, "gray")
    ax2.axis('off')
    if sino_comp:
        ax1.set_title("GT sinogram")
        ax2.set_title("RealESRGAN output sinogram")
    else:
        ax1.set_title("Original image")
        ax2.set_title("Reconstructed image")
    fig.text(.5, .05, result, ha='center')

    if bestcase:
        plt.savefig(str(recon_file)+"_bestcase_comparison.png")
    else:
        plt.savefig(str(recon_file)+"_comparison.png")
    #plt.show()

    print(result)


results_csv = recon_dir + "metrics_testing_images_realesrgan_recon"
if bestcase:
    results_csv = results_csv + "_bestcase.csv"
else:
    results_csv = results_csv + ".csv"

with open(results_csv, 'w', encoding='utf8', newline='') as output_file:
    fc = csv.DictWriter(output_file, fieldnames=results[0].keys())
    fc.writeheader()
    fc.writerows(results)
