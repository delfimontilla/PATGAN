import numpy as np
from pathlib import Path
from scipy.io import loadmat
from pydicom import dcmread 
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
    sino = np.array([sino])
    return sino


ANALYSIS_TYPE = "IMAGES"
bestcase = True
save_examples = True

recon_dir = "/PAT_GAN/results/res_pretrained_recon/"
recon_files = list(Path(recon_dir).glob("*.mat"))
recon_files = natsorted(recon_files)

if ANALYSIS_TYPE == "IMAGES":
    if bestcase:
        gt_dir = "/PAT_GAN/datasets/breast_phantoms/test/test_gt_recon/"
        load_path = Path(gt_dir)
        gt_files = list(load_path.glob("*.mat"))
        gt_files = natsorted(gt_files)
    else:
        images_paths_txt = "/PAT_GAN/scripts/data_curation/curated_test.txt"
        with open(images_paths_txt, 'r') as fp:
            gt_files = fp.readlines()
            gt_files = natsorted(gt_files)

elif ANALYSIS_TYPE == "SINOGRAMS":
    gt_dir = "/PAT_GAN/datasets/breast_phantoms/test/test_gt/"
    load_path = Path(gt_dir)
    gt_files = list(load_path.glob("*.mat"))
    gt_files = natsorted(gt_files)

if len(gt_files) != len(recon_files):
    raise Exception("Error: Number of files in gt and recon directories are not equal")


results = []
i = 0

for gt_file, recon_file in zip(gt_files, recon_files):
    filename = Path(recon_file).stem
    if ANALYSIS_TYPE == "IMAGES" and not bestcase:
        orig_img = dcmread(gt_file[:-1])
        orig = orig_img.pixel_array
        image = Image.fromarray(orig)
        image_re = image.resize((201,201))
        orig = np.array(image_re)
        orig = orig.astype(np.float64)
        orig = orig/orig.max()
    else:
        image = imfrommat(gt_file)
        orig = image[0].astype(np.float64)
        orig = orig/orig.max()

    recon = imfrommat(recon_file)
    recon = recon[0].astype(np.float64)
    if recon.max() != 0:
        norm_recon = recon/recon.max()
    else:
        norm_recon = recon

    SSIM = ssim(orig, norm_recon)
    RMSE = np.sqrt(mse(orig, norm_recon))
    PSNR = psnr(orig, norm_recon, data_range=1)
    PC = pearsonr(orig.flatten(), norm_recon.flatten())
    result = {"file": filename, "PC": PC[0], "ssmval": SSIM, "rmse": RMSE, "peaksnr": PSNR}
    results.append(result)

    if save_examples and i < 10:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,7))
            ax1.imshow(orig, "gray")
            ax1.axis('off')
            ax2.imshow(norm_recon, "gray")
            ax2.axis('off')
            if ANALYSIS_TYPE == "SINOGRAMS":
                ax1.set_title("GT sinogram")
                ax2.set_title("Sinogram")
            elif ANALYSIS_TYPE == "IMAGES":
                ax1.set_title("Original image")
                ax2.set_title("Reconstructed image")
            fig.text(.5, .05, result, ha='center')

            if bestcase:
                plt.savefig(str(recon_file)+"_bestcase_comparison.png")
            else:
                plt.savefig(str(recon_file)+"_comparison.png")
            #plt.show()

    print(result)
    i += 1


results_csv = recon_dir + f"results_{ANALYSIS_TYPE}{'bestcase' if bestcase else ''}.csv"

with open(results_csv, 'w', encoding='utf8', newline='') as output_file:
    fc = csv.DictWriter(output_file, fieldnames=results[0].keys())
    fc.writeheader()
    fc.writerows(results)
