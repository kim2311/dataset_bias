from PIL import Image
import glob
import os

root = '/local/a/ksivaman/dataset-bias/pascalVOCtoCAM2/testB'
rootgrey = '/local/a/ksivaman/dataset-bias/greyimgs2/'
for f in glob.glob(root + '/*.jpg'):
    im = Image.open(f).convert('L')
    k = f.lstrip('/local/a/ksivaman/dataset-bias/pascalVOCtoCAM2/testB/')
    im.save(rootgrey + k)
