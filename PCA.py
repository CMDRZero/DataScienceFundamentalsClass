from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import cv2
import numpy
import pickle as pk
import os
import numpy

pcahash = 'AC'

SCALE = 4

def DownScale(img):
    return img[::SCALE, ::SCALE]

def UpScale(img):
    return numpy.kron(img, numpy.ones((SCALE, SCALE, 1)))

def Compress(img):
    return pca.transform(img.flatten().reshape((1,)+img.flatten().shape))[0]

def DeCompress(img):
    return pca.inverse_transform(img.flatten().reshape((1,)+img.shape))[0].reshape(imshape)

imshape = DownScale(cv2.imread('frames/frame0.jpg')).shape

if os.path.exists(f'pca-{pcahash}.pk'):
   with open(f'pca-{pcahash}.pk', 'rb') as f:
        pca = pk.load(f)
else:
    assert __name__ == '__main__', f'Could not find PCA model'
    imgs = []
    print(f'Begin Img Fetch')
    for i in range(0, 5165, 5):
        img = cv2.imread(f'frames/frame{i}.jpg')
        imgs.append(DownScale(img))
    imshape = imgs[0].shape
    print(f'Image shape is {imshape}')
    print(f'Images fetched')


    X = numpy.array([img.flatten() for img in imgs])

    ##X = X[:, numpy.std(X, axis=0) > 0]

    #X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=64)
    pca.fit(X)
    with open(f'pca-{pcahash}.pk', 'wb') as f:
        pk.dump(pca, f)
    ##X_pca = pca.fit_transform(X)
