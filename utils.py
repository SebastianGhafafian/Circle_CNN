import numpy as np
import pandas as pd
import scipy as sc
from PIL import Image, ImageOps
import csv
import os
import shutil
from torch import randn
from torchvision.io import read_image
from torch.utils.data import Dataset
def make_archive(source, destination):
    """
    creates a .zip file from source directory to destination
    Example:
    source = /your/../path/
    destination = /your/../other/path/zipped_data.zip
    """   
    base = os.path.basename(destination)
    name = base.split('.')[0]
    format = base.split('.')[1]
    archive_from = os.path.dirname(source)
    archive_to = os.path.basename(source.strip(os.sep))
    shutil.make_archive(name, format, archive_from, archive_to)
    shutil.move('%s.%s'%(name,format), destination)

def circle(input_array,r,value):
    """
    -adds value to an input array within a radius r around a random center position
    -array is normalized to values between [0,255]
    Inputs:
    input_array: 2D numpy array of arbitrary size with dimensions larger than the radius
    r: radius of circle
    value: value that is added to array
    Outputs:
    output_array: normalized array with added circle
    x0: coordinate of circle center in dim 0
    y0: coordinate of circle center in dim 0

    """
    margin = r #margin from image boundary to center of circle
    output_array = np.zeros_like(input_array) #init output array
    #randomly select coordinates for center such that circle is fully contained in image
    x0 = np.random.randint(margin-1,output_array.shape[0]-(margin))
    y0 = np.random.randint(margin-1,output_array.shape[1]-(margin))
    # x0 = np.random.randint(margin-1,output_array.shape[0]-(margin+1)))
    # y0 = np.random.randint(margin-1,output_array.shape[1]-(margin+1)))
    #add value within radius of center
    for i in range(output_array.shape[0]):
        for j in range(output_array.shape[1]):
            if (x0 - i) ** 2 + (y0 - j) ** 2 <= r ** 2:
                output_array[i,j]= input_array[i,j] + value
    output_array = output_array / np.max(output_array)*255
    return output_array, x0, y0


def create_images(path,n,dim=[100,100],r_max=15):
    """
    creates image data of a circle in an array of specified dimensions
    Input:
    path: path to location of created data set
    n: number of images
    dim: images size (dimensions)
    r_max: maximum radius of circle (randomly chose between [5,r_max]) 
    Output:
    image data set of n images
    """
    cwd = os.getcwd()
    print(cwd)
    image_path = os.path.join(cwd, path)
    annotations = []
    Z = np.zeros(dim)
    for i in range(n):
        radius = np.random.randint(5,r_max)
        #radius = np.round(np.random.uniform(5,r_max)) #uniformly pick a radius between 5 and rad_max
        C,x0,y0 = circle(Z,radius,value=1) #create a white circle on black background
        annotations.append([f"img{i}.png",x0,y0,radius]) #write annotations with circle position and size
        #transform data into grayscale image
        img = Image.fromarray(C)
        img = ImageOps.grayscale(img)
        img.save(f"{image_path}/img{i}.png") #safe image
        if i % 100 == 0:
            print(f"Job {i/n * 100:.2f} % done.") #print status of operation

    #writing annotation file
    print("Writing Annotation.csv")
    with open(f"{image_path}/annotations.csv", 'w') as f:
        write = csv.writer(f)
        write.writerow(["image name", "x0", "y0","r"])
        write.writerows(annotations)

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
    
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file) # get labels
        self.img_dir = img_dir # store image directory
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        xlabel = self.img_labels.iloc[idx, 1]
        ylabel = self.img_labels.iloc[idx, 2]
        radius = self.img_labels.iloc[idx, 3]
        # apply input transformation
        if self.transform:
            image = self.transform(image)
        # apply target transformation
        if self.target_transform:
            xlabel = self.target_transform(xlabel)
            ylabel = self.target_transform(ylabel)
            radius = self.target_transform(radius)

        return image, np.array([xlabel,ylabel,radius],dtype=np.float32) ####how to I work with