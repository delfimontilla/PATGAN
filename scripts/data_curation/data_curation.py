from pydicom import dcmread
from pathlib import Path
import numpy as np
import copy
from scipy.stats import mstats
import random
import os 


orig_dataset = Path("/PAT_GAN/datasets/original_dataset")
orig_filenames = list(orig_dataset.glob("**/*.dcm"))
curated_filenames = copy.deepcopy(orig_filenames)


print(len(orig_filenames))

#%%

for i, file in enumerate(orig_filenames):
    data = dcmread(file)
    img = data.pixel_array
    if np.count_nonzero(img)<=60000:
        print(i)
        curated_filenames.remove(file)
        continue

with open(r'curated_nonzero.txt', 'w') as fp:
    for item in curated_filenames:
        fp.write("%s\n" % str(item))
    print('Done')

#%%

pearson = []
with open('curated_nonzero_pearson8.txt', 'r') as fp:
    f = fp.readlines()
    fname = "aaa"
    inic=114
    curated_filenames = copy.deepcopy(f)
    for i, file0 in enumerate(f[inic:]):
        if fname == Path(file0).stem[:-4]:
            continue
        fname = Path(file0).stem[:-4]
        data0 = dcmread(str(file0[:-1]))
        img0 = data0.pixel_array
        print(i)
        for j, file in enumerate(f[inic+i+1:]):
            data = dcmread(file[:-1])
            img = data.pixel_array
            p = mstats.pearsonr(img0,img)
            if p[0] > 0.5:
                f.remove(file)
                curated_filenames.remove(file)

with open('curated_nonzero_pearson9.txt', 'w') as fp:
    for item in curated_filenames:
        fp.write("%s" % str(item))
    print('Done')


#%%

with open('curated_nonzero_pearson9.txt', 'r') as fp:
    f = fp.readlines()
    random.seed(10)
    random.shuffle(f)
    with open('curated_train.txt', 'w') as file:
        for item in f[:1500]:
            file.write("%s" % str(item))
    with open('curated_val.txt', 'w') as file:
        for item in f[1500:1900]:
            file.write("%s" % str(item))
    with open('curated_test.txt', 'w') as file:
        for item in f[1900:]:
            file.write("%s" % str(item))
    print('Done')


#%%
with open('curated_val.txt', 'r') as fp:
    f = fp.readlines()
    f.sort()
    with open('curated_val_relpath.txt', 'w') as file:
        for item in f:
            s = item.split("/")
            item = os.path.join(s[-2],s[-1])
            file.write("%s" % str(item))