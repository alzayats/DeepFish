print('START')


'''`imread` is deprecated in SciPy 1.0.0, and will be removed in 1.2.0.
Use ``imageio.imread`` instead.
  mask = scipy.misc.imread(fname)
`imsave` is deprecated in SciPy 1.0.0, and will be removed in 1.2.0.
Use ``imageio.imwrite`` instead.
  scipy.misc.imsave(out_path2 + '/' + file, pointss)'''
"""import imageio
imageio.imread()
imageio.imwrite()"""
##################################################################

'''dot_2_circle'''
from skimage.draw import circle
import numpy as np
def dot_2_circle(y):
   ind = np.where(y != 0)
   N_CLASSES = 1
   nr = y.shape[0]
   nc = y.shape[1]
   blck_img = np.zeros([nr, nc, N_CLASSES], dtype=np.float64)
   for r, c in zip(ind[0], ind[1]):
       rr, cc = circle(r, c, 5, shape=(nr,nc )) # shape which is used to determine the maximum extent of output pixel coordinates.
       blck_img[rr, cc] = 1
   y = blck_img
   return np.squeeze(y)


'''circle_2_dot'''
import skimage.io
from skimage import measure
from skimage.draw import circle
def circle_2_dot(y):
   N_CLASSES = 1
   nr = y.shape[0]
   nc = y.shape[1]
   blck_img = np.zeros([nr, nc, N_CLASSES], dtype=np.float64)
   # make binary
   bw = y.copy()
   bw[bw >= 1] = 1
   bw[bw < 1] = 0
   bw = np.asarray(bw, dtype=bool)
   labels = measure.label(bw)
   for region in measure.regionprops(labels):
       minr, minc, maxr, maxc = region.bbox
       r = (minr + maxr) / 2.
       c = (minc + maxc) / 2.
       blck_img[int(r), int(c)] = 1
   y = blck_img
   return np.squeeze(y)


'''circle_2_circle'''
import skimage.io
from skimage import measure
from skimage.draw import circle
def circle_2_circle(y):
   N_CLASSES = 1
   nr = y.shape[0]
   nc = y.shape[1]
   blck_img = np.zeros([nr, nc, N_CLASSES], dtype=np.float64)
   # make binary
   bw = y.copy()
   bw[bw >= 1] = 1
   bw[bw < 1] = 0
   bw = np.asarray(bw, dtype=bool)
   labels = measure.label(bw)
   for region in measure.regionprops(labels):
       minr, minc, maxr, maxc = region.bbox
       r = (minr + maxr) / 2.
       c = (minc + maxc) / 2.
       rr, cc = circle(r, c, 15, shape=(nr,nc )) # shape which is used to determine the maximum extent of output pixel coordinates.
       blck_img[rr, cc] = 1
   y = blck_img
   return np.squeeze(y)


'''count the points in image'''
import scipy.misc
import numpy as np
def count_points(y):
    # points = scipy.misc.imread(path)
    points = y
    ind = np.where(points != 0)
    # print(len(ind[0]))
    return len(ind[0])



'''blobs_2_count'''
# the blobs_2_count function takes an image and returns the number of counts
# set morph=True if you need to do morphology watershed
############todo test it
# mask = scipy.misc.imread(image path) # todo image path
# print(blobs_2_count(mask, morph=True))
###################
import cv2
import numpy as np
import scipy.misc
from skimage.filters import sobel
from skimage import io, morphology, measure
def blobs_2_count(mask, morph=True):  #todo set morph=True to use watershed
   if morph:
       mask = morph_dots(mask)
   bw = mask.copy()
   bw[bw >= 1] = 1
   bw[bw < 1] = 0
   bw = np.asarray(bw, dtype=bool)
   labels = measure.label(bw)
   count = len([region for region in measure.regionprops(labels)])
   return count

def morph_dots(mask):
   elevation_map = sobel(mask)
   markers = np.zeros_like(mask)
   markers[mask < 1] = 1
   markers[mask > 100] = 2
   mask = morphology.watershed(elevation_map, markers)
   mask = scipy.misc.toimage(mask)
   mask = scipy.misc.fromimage(mask)
   return mask
###################
###################



'''convert RGB image to binary image'''
'''cyclone AI from red to white'''
import scipy
import numpy as np
from os import listdir
from tqdm import tqdm
def convert_RGBmask_2_binary(inpath, outpath):
    for file in tqdm(listdir(inpath)):
        filename, extension = splitext(file)
        try:
            if extension in ['.png']:
                image = scipy.misc.imread(inpath+'/'+file, mode='L').clip(0, 1)  #‘L’ (8-bit pixels, black and white)
                scipy.misc.imsave(outpath+'/'+file, np.squeeze(image*255))
        except OSError:
           print('Cannot convert %s' % file)



def check_if_binary(image):
    print('test if 0&1','='*20)
    image2 = scipy.misc.imread(image).clip(0, 1)
    print(np.where(image2 > 1))
    print(image2[500:600,700:800])  # this is x&y of the blob need to test




'''Convert jpg to png'''
from PIL import Image
from os import listdir
from os.path import splitext
from tqdm import tqdm
def Convert_jpg_2_png(inpath, outpath, target ='.png' ):
    for file in tqdm(listdir(inpath)):
       filename, extension = splitext(file)
       try:
           if extension in ['.jpg', '.JPG']:
               im = Image.open(inpath+'/' + filename + extension)
               im.save(outpath+'/' + filename + target)
       except OSError:
           print('Cannot convert %s' % file)

'''Convert png to jpg'''
from PIL import Image
from os import listdir
from os.path import splitext
from tqdm import tqdm
def Convert_png_2_jpg(inpath, outpath, target ='.jpg' ):
    for file in tqdm(listdir(inpath)):
       filename, extension = splitext(file)
       try:
           if extension in ['.png', '.PNG']:
               im = Image.open(inpath+'/' + filename + extension)
               im.save(outpath+'/' + filename + target)
       except OSError:
           print('Cannot convert %s' % file)


"""jpg_png_multiply"""
def jpg_png_multiply(jpgpath, pngpath, outpath):
    for file_in in tqdm(os.listdir(os.path.join(pngpath))):
        file = os.path.splitext(file_in)[0]
        x = cv2.imread('{}{}.JPG'.format(jpgpath, file))
        y = cv2.imread('{}{}.png'.format(pngpath, file))
        image_3 = cv2.addWeighted(x, 0.3, y, 0.5, 0)
        scipy.misc.imsave('{}{}.jpg'.format(outpath, file),image_3)

''' remove (.mask.0) from filename'''
import os
from tqdm import tqdm
def remove_LASTpartOF_filename(inpath, outpath, part, extensions = '.png'):
    for file in tqdm(os.listdir(inpath)):
        try:
            _, extension = splitext(file)
            if extension in [extensions]:
                filename, rmv_txt = file.split(part)
                src = inpath + '/' + file
                dst = outpath + '/' + filename + extension
                # print(file, filename,extension, rmv_txt)
                os.rename(src, dst)
        except OSError:
            print('Cannot Rename %s' % file)

''' remove FIRST part OF filename'''
import os
from tqdm import tqdm
def remove_FIRSTpartOF_filename(inpath, outpath, part):
    for file in tqdm(os.listdir(inpath)):
        try:
            _, extension = splitext(file)
            if extension in ['.jpg', '.png']:
                rmv_txt, filename,  = file.split(part)
                src = inpath + '/' + file
                dst = outpath + '/' + filename
                # print(file, filename,extension, rmv_txt , dst)
                os.rename(src, dst)
        except OSError:
            print('Cannot Rename %s' % file)

''' (rename & copy & move) from dataframe'''
import pandas as pd
from tqdm import tqdm
import shutil
def rename_copy_move_fromDF(inpath, outpath, DFpath):
    img_name=pd.read_csv(DFpath)
    img_name_lst =list(img_name.num_name)
    for file in tqdm(img_name_lst):
       _,name = file.split('-')
       try:
           srcc = inpath + '/' + name
           dstt =outpath + '/' + file
           # print(srcc,dstt)
           # shutil.copyfile(srcc, dstt)
           # shutil.move(srcc, dstt)
       except:
           print('Cannot Rename %s' % file)


'''rename two columns in a pandas dataframe'''
import pandas as pd
def rename_columns_fromDF(DFinpath, DFoutpath):
    org_DF=pd.read_csv(DFinpath)
    num_name = []
    for name, count in zip(org_DF[' image name'],org_DF[' num of fishes']):
       print(name)
       print(name[6:])
       print(count)
       cn  = str(count) + '_' + name[6:]
       print(cn)
       num_name.append(cn)
    org_DF['num_name'] = num_name
    org_DF.to_csv(DFoutpath)


'''change all circle_2_dot'''
def all_circle_2_dot():
    path = "D:/Datasets/JCU_FISH/jcu_fish_train_yes_step10_valid_png/binry_mask/"
    out_path2 = 'D:/Datasets/JCU_FISH/jcu_fish_train_yes_step10_valid_png/points/'
    for file in tqdm(listdir(path)):
        fname = path + file
        mask = scipy.misc.imread(fname)
        pointss = circle_2_dot(mask)
        scipy.misc.imsave(out_path2 + '/' + file, pointss)

'''jcu_fish_dataset_2_png'''
def jcu_fish_dataset_2_png():
    path = "H:/datasets/fish/jcu_fish_dataset/train_yes_step10/"
    out_path = 'H:/datasets/fish/jcu_fish_dataset/train_yes_step10_valid_png/'
    for file in listdir(path):
        in_path = path + file + '/valid/'
        Convert_jpg_2_png(in_path, out_path)


''' paste image on black background'''
def paste_on_background (image_in, image_out, bg_size=(640, 640)):
   from PIL import Image
   img = Image.open(image_in)
   img_w, img_h = img.size
   background = Image.new('RGB', bg_size)
   bg_w, bg_h = background.size
   offset = ((bg_w - img_w) // 2, (bg_h - img_h) // 2)
   background.paste(img, offset)
   background.save(image_out)

''' create black mask for imgAnnotation.exe '''
def imgAnnotationCV2():
    from shutil import copyfile
    src = "D:/Datasets/jcu_fish_train_yes_step10_valid_png/7482_F1_f000000.mask.0.png"
    path = "D:/Datasets/jcu_fish_train_yes_step10_valid_png/temp/"
    for file in tqdm(listdir(path)):
        img_name , ext =splitext(file)
        dst = path+ '/' + img_name + '.mask.0.png'
        if os.path.isfile(dst):
            print(dst)
            continue
        copyfile(src, dst)

"""create black image"""
def create_black_image():
   import imageio
   in_path = "D:/Datasets/JCU_FISH/jcu_fish_step10_1617/images/empty/"
   out_path = "D:/Datasets/JCU_FISH/jcu_fish_step10_1617/masks/empty/"
   for file in tqdm(listdir(in_path)):
       img_name , ext =splitext(file)
       image_in = in_path + file
       image_out = out_path + img_name + '.png'
       # img = imageio.imread(image_in, format='L')
       img = imageio.imread(image_in)
       blck_img = np.zeros(img.shape[:2], dtype=np.uint8)
       # blck_img = img.copy() * 0
       # mask = Image.new('RGB', img.size)
       imageio.imwrite(image_out, blck_img)


'''##################simple multiply with addWeighted ###########'''
def image_multiply():
    from tqdm import tqdm
    import os
    import scipy.misc
    import cv2
    dir_x = "D:/Datasets/JCU_FISH/jcu_fish_segmentation_step10_/temp/jpg/"
    dir_y = "D:/Datasets/JCU_FISH/jcu_fish_segmentation_step10_/temp/delete/old_masks/"
    dir_save = "D:/Datasets/JCU_FISH/jcu_fish_segmentation_step10_/temp/img_multiply/"
    for file_in in tqdm(os.listdir(os.path.join(dir_y))):
        try:
           # file = os.path.splitext(file_in)[0]
           file, rmv_txt,  = file_in.split('.m')
           x = cv2.imread('{}{}.JPG'.format(dir_x, file))
           y = cv2.imread('{}{}.mask.0.png'.format(dir_y, file))
           image_3 = cv2.addWeighted(x, 0.9, y, 0.3, 0)
           scipy.misc.imsave('{}{}.jpg'.format(dir_save, file),image_3)
        except:
            print('Cannot Rename %s' % file_in)

'''#########################################  video good ###############################'''
def make_video():
    import cv2
    import os
    from tqdm import tqdm
    image_folder = 'D:/Datasets/JCU_FISH/jcu_fish_segmentation_step10_/temp/img_multiply/'
    video_name = 'D:/Datasets/JCU_FISH/jcu_fish_segmentation_step10_/temp//seg_video2.mp4'
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    width, height =width//2,height//2
    # video = cv2.VideoWriter(video_name, -1, 1, (width,height))
    # video = cv2.VideoWriter(video_name, -1, 10, (500,500))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
    # video = cv2.VideoWriter(video_name, fourcc , 9, (500,500))
    video = cv2.VideoWriter(video_name, fourcc , 9, (width,height))
    for image in tqdm(images):
        video.write(cv2.resize((cv2.imread(os.path.join(image_folder, image))), (width,height), interpolation=cv2.INTER_LINEAR))
        # video.write(cv2.imread(os.path.join(image_folder, image)))
    # cv2.destroyAllWindows()
    video.release()



'''visualize_dots'''
def visualize_dots(img, points, save_path, size=5):
    import cv2
    import imageio
    img = imageio.imread(img)
    points = imageio.imread(points).clip(0, 1)
    points = points.squeeze()
    gt_count = np.count_nonzero(points)
    y,x= np.where(points==1) #Get locations of 1s
    overlay = img.copy()
    img_overlay =overlay
    for x_cent, y_cent in zip(x,y):
        img_overlay = cv2.circle(overlay, (x_cent,y_cent), size, (0,0,255), -1) #Draw a filled circle at these locations
    alpha = 0.4  # Transparency factor.
    # Following line overlays transparent circle over the image
    img = cv2.addWeighted(img_overlay, alpha, img, 1 - alpha, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, f'gt_count={gt_count}', (10, 50), font, 2, (0, 0, 0), 5, cv2.LINE_AA)
    imageio.imwrite(save_path, img)

'''all_visualize_dots'''
def all_visualize_dots():
    from os.path import splitext
    imgs_path = 'D:/Datasets/JCU_FISH/jcu_fish_step10_1617/images/valid/'
    points_path = 'D:/Datasets/JCU_FISH/jcu_fish_step10_1617/masks/valid/'
    root_path = 'D:/Datasets/JCU_FISH/jcu_fish_train_yes_step10_valid_png/visualize_dots/'
    for file in tqdm(listdir(imgs_path)):
        img_path = imgs_path + file
        point_path = points_path + splitext(file)[0] + '.png'
        save_path = root_path + 'dots_' +file
        visualize_dots(img_path, point_path, save_path, size=10)

def train_test_df_JCU():
    from pandas import DataFrame
    import imageio
    imgs_path = 'D:/Datasets/JCU_FISH/jcu_fish_step10_1617/masks/empty/'
    image_names = []
    gt_counts = []
    for file in tqdm(listdir(imgs_path)):
        img_path = imgs_path + file
        image_name = splitext(file)[0]
        points = imageio.imread(img_path).squeeze().clip(0, 1)
        gt_count = np.count_nonzero(points)
        image_names.append(image_name)
        gt_counts.append(gt_count)
        # gt_count2 = [int(points.sum())]
    image_dict = {'ID':image_names, 'count':gt_counts}
    image_df = DataFrame(image_dict, columns=['ID', 'count'])
    image_df.to_csv('D:/Datasets/JCU_FISH/jcu_fish_step10_1617/ALL_empty_images_dataframe.csv', index=None, header=True)

def save_train_test_DataFrame():
    import pandas as pd
    dataframe_dir1 = 'D:/Datasets/JCU_FISH/jcu_fish_step10_1617/ALL_valid_images_dataframe.csv'
    dataframe_dir2 = 'D:/Datasets/JCU_FISH/jcu_fish_step10_1617/ALL_empty_images_dataframe.csv'
    data_dir = 'D:/Datasets/JCU_FISH/jcu_fish_step10_1617/'
    df1 = pd.read_csv(dataframe_dir1)
    df2 = pd.read_csv(dataframe_dir2)
    df_image_FC = pd.concat([df1, df2], axis=0)
    df_image_FC = df_image_FC.sample(frac=1)
    train_DF = df_image_FC[:1200]
    val_DF = df_image_FC[1200: 1700]
    test_DF = df_image_FC[1700:]
    train_DF.to_csv(os.path.join(data_dir, 'train.csv'), index=None, header=True)
    val_DF.to_csv(os.path.join(data_dir, 'val.csv'), index=None, header=True)
    test_DF.to_csv(os.path.join(data_dir, 'test.csv'), index=None, header=True)


def print_score_list():
    import pandas as pd
    import pickle
    # root = 'D:/prototypes/underwater_fish/fish_glance/a1229be27745f92a7c9bc2cfbdee8e4c/'
    root = 'D:/prototypes/underwater_fish/fish_lcfcn/f92a7e2bb807bca700c5545586b26cda/'
    score_pkl = root + 'score_list.pkl'
    with open(score_pkl, "rb") as f:
        score_list = pickle.load(f)
    score_df = pd.DataFrame(score_list)
    score_df.to_csv((root+'score_list.csv'), index=None, header=True)
    print("\n", score_df[["epoch", "train_loss", "val_score"]], "\n")

''' JCUfish_ subdir as label dataframe'''
def subdir2label_dataframe():
   import os
   from glob import glob
   import pandas as pd
   from tqdm import tqdm
   files = []
   df_image_subdir = []
   # start_dir = os.path.abspath('D:/Datasets/JCU_FISH/JCU_fish_classification_40K_images/train_yes_step10/')
   start_dir = os.path.abspath('D:\Datasets\JCU_FISH\jcu_fish_step10_1617\images/')
   saveDF_dir = 'D:/Datasets/JCU_FISH/'
   pattern = "*.jpg"
   for dir, _, _ in tqdm(os.walk(start_dir)):
       files.extend(glob(os.path.join(dir, pattern)))
   for path in tqdm(files):
       tlabel = os.path.split(path)[0]
       label = os.path.split(tlabel)[-1]
       id = os.path.split(path)[-1]
       # type = tlabel.split("\\")[-2]
       type = os.path.splitext(id)[0][:4]
       if path.endswith('.jpg'):
           df_image_subdir.append([type, id, label])
   df_image_subdir = pd.DataFrame(df_image_subdir, columns=["type","id",'vld_empty'])
   print(df_image_subdir.head())
   df_image_subdir.to_csv(saveDF_dir + 'imagesdot.csv', index=False)


""" Make new column in Panda dataframe by adding values from other columns """
def splitext_df(in_path, out_path):
    DF=pd.read_csv(in_path)
    DF['habitat'] = [os.path.splitext(x)[0][:4] for x in DF['ID']]
    print(DF.head())
    DF.to_csv(out_path + 'ALL_val.csv', index=False)


def statistics_from_df(in_path, out_path):
    DF=pd.read_csv(in_path)
    DF['type'] = [os.path.splitext(x)[0][:4] for x in DF['ID']]
    # Renaming grouped statistics from groupby operations
    DF2 = DF.groupby('type').count()
    DF = DF.groupby('type').agg({"count": [min, max, sum]})
    # Using ravel, and a string join, we can create better names for the columns:
    DF.columns = ["_".join(x) for x in DF.columns.ravel()]
    DF3 = pd.concat([DF,DF2], axis=1)
    print(DF)
    print(DF2)
    print(DF3)
    DF3.to_csv(out_path + 'ALL_valid_images_statistics.csv')

Habitats_list = \
["7117",
"7393",
"7398",
"7426",
"7434",
"7463",
"7482",
"7490",
"7585",
"7623",
"9852",
"9862",
"9866",
"9870",
"9892",
"9894",
"9898",
"9907",
"9908"]

Habitats_dict = \
{"7117":	"Rocky Mangrove - prop roots"
,"7268":	"Sparse algal bed"
,"7393":	"Upper Mangrove – medium Rhizophora"
,"7398":	"Sandy mangrove prop roots"
,"7426":	"Complex reef"
,"7434":	"Low algal bed"
,"7463":	"Seagrass bed"
,"7482":	"Low complexity reef"
,"7490":	"Boulders"
,"7585":	"Mixed substratum mangrove - prop roots"
,"7623":	"Reef trench"
,"9852":	"Upper mangrove - tall rhizophora"
,"9862":	"Large boulder"
,"9866":	"Muddy mangrove - pneumatophores and trunk"
,"9870":	"Muddy mangrove – pneumatophores"
,"9892":	"Bare substratum"
,"9894":	"Mangrove - mixed pneumatophore prop root"
,"9898":	"Rocky mangrove - large boulder and trunk"
,"9907":	"Rock shelf"
,"9908":	"Large boulder and pneumatophores"}

''' test the code'''
# in_path = 'D:/Datasets/JCU_FISH/jcu_fish_train_yes_step10_valid_png/binry_mask/'
# out_path = 'D:/Datasets/JCU_FISH/jcu_fish_train_yes_step10_valid_png/binry_mask2/'
in_path = 'D:\Datasets\JCU_FISH\jcu_fish_counting_step10_1617/val.csv'
out_path = 'D:\Datasets\JCU_FISH\jcu_fish_counting_step10_1617/'
# jpgpath = 'H:\Cyclone_AI\Segmented_images\segment_for_size_100\images/'
# pngpath = 'H:\Cyclone_AI\Segmented_images\segment_for_size_100\masks/'
# outpath = 'H:\Cyclone_AI\Segmented_images\segment_for_size_100\jpg_png_multi/'
# # in_path2 = 'F:/datasets/Simone_dataset/2013_count_train/png/rgb_mask'
# out_path2 = 'D:/Datasets/FishCount_annotated//binary'
# in_path3 = 'F:/datasets/Simone_dataset/2013_count_train/jpg_png'
# out_path3 = 'F:/datasets/Simone_dataset/2013_count_train/jpg_png'
# DFpath = 'F:/datasets/Simone_dataset/trainingSetImageInfo_name_count2.csv'
# DFinpath = 'F:/datasets/Simone_dataset/trainingSetImageInfo.csv'
# DFoutpath = 'F:/datasets/Simone_dataset/trainingSetImageInfo2.csv'
# path = out_path2 + '/orgn.png'

# Convert_jpg_2_png(in_path, out_path)
# Convert_png_2_jpg(in_path, out_path)
# jpg_png_multiply(jpgpath, pngpath, outpath)
# convert_RGBmask_2_binary(in_path, out_path)
# print(blobs_2_count(mask, morph=True))
# pointss = circle_2_dot(mask)
# print(count_points(pointss))
# circless = dot_2_circle(mask)
# bigcircless = circle_2_circle(mask)
# scipy.misc.imsave(out_path2+'/'+'circless.png', circless )
# scipy.misc.imsave(out_path2+'/'+'pointss.png', pointss )
# scipy.misc.imsave(out_path2+'/'+'bigcircless.png', bigcircless )
# print(blobs_2_count(circless, morph=True))

# remove_LASTpartOF_filename(in_path, out_path, '.m')
# remove_FIRSTpartOF_filename(in_path3, out_path3,  '-')
# rename_copy_move_fromDF(in_path3, in_path3,DFpath )
# rename_columns_fromDF(DFinpath, DFoutpath)
# jcu_fish_dataset_2_png()
# imgAnnotationCV2()
# image_multiply()
# convert_RGBmask_2_binary(in_path, out_path)
# remove_LASTpartOF_filename(in_path, out_path, '.m', extensions = '.png')
# all_circle_2_dot()
# import imageio
# imgg = 'D:\Datasets\JCU_FISH\jcu_fish_step10_1617\masks\empty/7393_NF2_f000080.png'
# img1 = cv2.imread(imgg)
# img2 = scipy.misc.imread(imgg)
# img3 = imageio.imread(imgg)
# print(img1,img2, img3)

# all_visualize_dots()
# make_video()
# create_black_image()
# train_test_df_JCU()
# save_train_test_DataFrame()
# print_score_list()
# subdir2label_dataframe()
# splitext_df(in_path, out_path)
# statistics_from_df(in_path, out_path)

print('END')



