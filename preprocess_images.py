from PIL import Image
import multiprocessing
#from multiprocessing import Pool
import glob
import argparse
import tqdm
from os.path import exists
import torch
import os
import torchvision.transforms as transforms

trans = transforms.Compose([transforms.Resize(256),
                            transforms.CenterCrop([224,224])])

parser = argparse.ArgumentParser(description='crop out objectnet red border')
parser.add_argument('src', type=str,
                    help='images source folder')
parser.add_argument('dst', type=str,
                    help='images destination folder')

args = parser.parse_args()
print(args)


def crop(filename):
    file = filename.split("/")[-1]
    object_type = filename.split("/")[-2]
    try:
        if exists(f"{args.dst}{object_type}/{file}"):
            return
        im = Image.open(filename)
        h, w = im.size
        #print "../../public/images_thumb/" + filename.split("/")[-1])
        #cropped_im = im.crop((2, 2, h-2, w-2))
        #cropped_im.save(args.dst+"/%s" % filename.split("/")[-1])
        cropped_im = trans(im)
        cropped_im.save(f"{args.dst}{object_type}/{file}")
        return 'OK'
    except Exception as e:
        print(e)
        raise e

images_classes=[x.split("/")[-1] for x in glob.glob(args.src+"*")]
pool = multiprocessing.Pool(multiprocessing.cpu_count())

for obj in images_classes:
    #imagesPaths=imagesPaths[:20]
    try:
        os.system("mkdir " + args.dst + obj)
    except:
        pass
    imagesPaths = glob.glob(args.src+obj+"/*")
    results = list(tqdm.tqdm(pool.imap(crop, imagesPaths), total=len(imagesPaths)))
    #crop(imagesPaths[0])

