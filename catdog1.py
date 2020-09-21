# https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-photos-of-dogs-and-cats/

from os import listdir
from os.path import join
from numpy import asarray
from numpy import save
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

folder = r'C:\data\dogs-vs-cats\train'

def get_photo(file):
    return img_to_array(load_img(join(folder,file),target_size=(200,200)))

def getlabel(file):
    return 1.0 if file.startswith('cat') else 0.0

photos_with_labels=[(get_photo(file),getlabel(file)) for file in listdir(folder)]

photos = asarray([photo for (photo,_) in photos_with_labels])

labels = asarray([labels for (_,label) in photos_with_labels])

save('photos.npy',photos)

save('labels.npy',labels)