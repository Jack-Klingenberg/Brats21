# Script to rewrite all MRI images to correct sizes 
# (180x180x180) for the VNet

import nibabel as nib
import os 
from scipy.ndimage import zoom
import shutil


if os.path.exists("../temp_data"):
    shutil.rmtree("../temp_data")
os.mkdir("../temp_data")

def resize_MRI_image(id):
    ext = ["flair", "t1ce", "t2", "t1", "seg"]
    resimages = []
    
    os.mkdir("../temp_data/" + id + "/")
    for extention in ext:
        pathhhh = "../data/" + id + "/" + id + "_" + extention +".nii.gz"
        if(not os.path.exists(pathhhh)):
            continue
        mri = nib.load(pathhhh)
        ima = mri.get_fdata()
        ima = zoom(ima,(180/ima.shape[0], 180/ima.shape[1], 180/ima.shape[2]))

        clipped_img = nib.Nifti1Image(ima, mri.affine)
        nib.save(clipped_img, "../temp_data/" + id+"/" +id + "_" + extention +".nii.gz")
        
    return

# Left off at id496
ids = [ ("BraTS2021_"+ '{:05d}'.format(i)) for i in range(0,1667)]

for i,id in enumerate(ids):
    resize_MRI_image(id)
    if(i%10==0):
        print("id [%i] resizing completed." %(i))

input("checkpoint - waiting to remove original data path...")

os.remove("../data")
os.rename("../temp_data", "../data")
