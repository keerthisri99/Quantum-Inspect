import cv2
import numpy as np
import os
from skimage.measure import shannon_entropy
from skimage.feature import graycomatrix,graycoprops
x=[]
y=[]
count=1
for file in os.listdir("dataset"):
    if(count<=200):
        path=os.path.join("dataset",file)
        img=cv2.imread(path)
        if img is None:
            print("invalid img")
            continue
        resized_img=cv2.resize(img,(128,128))
        gray_img=cv2.cvtColor(resized_img,cv2.COLOR_BGR2GRAY)
        norm_img=gray_img/255.0
        mean=np.mean(gray_img)
        std=np.std(gray_img)
        max=np.max(gray_img)
        min=np.min(gray_img)
        edges=cv2.Canny(gray_img,100,200)
        edge_count=np.count_nonzero(edges)
        edge_density=np.mean(edges>0)
        entropy=shannon_entropy(gray_img)
        contrast=np.percentile(gray_img,95)-np.percentile(gray_img,5)
        glcm=graycomatrix(gray_img,distances=[1],angles=[0],levels=256,symmetric=True,normed=True)
        glcm_homogeneity=graycoprops(glcm,'homogeneity')[0,0]
        glcm_contrast=graycoprops(glcm,'contrast')[0,0]
        features=[mean,std,max,min,edge_count,edge_density,entropy,contrast,glcm_homogeneity,glcm_contrast]
        if (count%2==1):
            x.append(features)
            y.append(0)
        else:
            x.append(features)
            y.append(1)
        count=count+1
X = np.array(x)
Y = np.array(y)

os.makedirs("data", exist_ok=True)

np.save("data/X.npy", X)
np.save("data/y.npy", Y)

print("Feature extraction completed!")
print("X shape:", X.shape)
print("Y shape:", Y.shape)
