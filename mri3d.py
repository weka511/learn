from pydicom           import dcmread
from os                import sep, listdir, walk
from os.path           import join, normpath
from pandas            import read_csv
from re                import match
from warnings          import warn


class MRI_Series:
    def __init__(self,name):
        self.name           = name
        self.missing_images = set()

    def add_images(self,filenames):
        def extract_digits(s):
            m = match(r'\D*(\d+)\D+',s)
            if m:
                return int(m.group(1))
        seqs = sorted([extract_digits(name) for name in filenames])
        self.N = seqs[-1]
        if len(seqs) + seqs[0] -1 == self.N: return
        self.missing_images = set([i+1 for i in range(seqs[-1]) if i+1 not in seqs])

    def __len__(self):
        return self.N

class MRI_Study:
    def __init__(self,name,path):
        self.series = None
        for dirpath, dirnames, filenames in walk(path):
            if self.series == None:
                self.series = {series_name: MRI_Series(series_name) for series_name in dirnames}
            else:
                path_components = normpath(dirpath).split(sep)
                series = self.series[path_components[-1]]
                series.add_images(filenames)
                if len(series.missing_images)>0:
                    warn(f'Study {name} {series.name} has images missing: {series.missing_images}')

class MRI_Dataset:
    def __init__(self,path,folder):
        self.studies = {name:MRI_Study(name,join(path,folder,name)) for name in listdir(join(path,folder))}

class Labelled_MRI_Dataset(MRI_Dataset):
    def __init__(self,path,folder,labels='train_labels.csv'):
        super().__init__(path,folder)
        self.labels = read_csv(join(path,labels),dtype={'BraTS21ID':str})


if __name__=='__main__':
    training = Labelled_MRI_Dataset(r'D:\data\rsna','train')
    test     = MRI_Dataset(r'D:\data\rsna','test')
