import kaggle
from os import system 
kaggle.api.authenticate()

def download(
    source      = 'train',
    file        = '000a6c98-bb9b-11e8-b2b9-ac1f6b6435d0_blue.png',
    competition = 'hpa-single-cell-image-classification',
    path        = 'c:\data\hpa-scc',
    colours = ['green','blue','red','yellow']):
    for colour in colours:
        f = f'{source}/{file}_{colour}.png'
        print(f)
        system(f'kaggle competitions download -f {f}  -c {competition} -p {path}')

if __name__=='__main__':
    with open(r'C:\data\hpa-scc\train.csv') as files:
        for line in files:
            download(file=line.split(',')[0])