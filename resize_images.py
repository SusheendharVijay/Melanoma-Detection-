import os 
import glob 
from tqdm import tqdm 
from PIL import Image, ImageFile 
from joblib import Parallel,delayed 

ImageFile.LOAD_TRUNCATED_IMAGES = True
TRAINING_DIR = "/home/sushi/code/Kaggle/Melanoma-Detection-/input/jpeg/train"
TESTING_DIR = "/home/sushi/code/Kaggle/Melanoma-Detection-/input/jpeg/test"
NEW_SIZE = 224




def resize_image(image_path,output_folder,resize):
    base_name = os.path.basename(image_path)
    outpath = os.path.join(output_folder, base_name)
    img = Image.open(image_path)
    img = img.resize((resize[1],resize[0]),resample=Image.BILINEAR)
    img.save(outpath)


input_folder = TRAINING_DIR
output_folder = "/home/sushi/code/Kaggle/Melanoma-Detection-/input/train"+str(NEW_SIZE)
images = glob.glob(os.path.join(input_folder,"*.jpg"))

Parallel(n_jobs=12)(
    delayed(resize_image)(
        i,output_folder,(NEW_SIZE,NEW_SIZE)
    ) for i in tqdm(images)
)


input_folder = TESTING_DIR
output_folder = "/home/sushi/code/Kaggle/Melanoma-Detection-/input/test"+str(NEW_SIZE)
images = glob.glob(os.path.join(input_folder,"*.jpg"))

Parallel(n_jobs=12)(
    delayed(resize_image)(
        i,output_folder,(NEW_SIZE,NEW_SIZE)
    ) for i in tqdm(images)
)








    

