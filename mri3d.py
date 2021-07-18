from matplotlib.pyplot import axes, cm, figure, savefig, show, title
from pydicom           import dcmread
from os                import sep, listdir, walk
from os.path           import join, normpath
from pandas            import read_csv
from re                import match
from warnings          import warn


class MRI_Series:

    def __init__(self,name):
        self.name             = name
        self.missing_images   = set()
        self.dirpath          = None
        self.image_plane      = None
        self.description      = None
        self.patient_position = None

    def add_images(self,dirpath,filenames):
        def extract_digits(s):
            m = match(r'\D*(\d+)\D+',s)
            if m:
                return int(m.group(1))
        self.dirpath = dirpath
        seqs         = sorted([extract_digits(name) for name in filenames])
        self.N       = seqs[-1]
        if len(seqs) + seqs[0] -1 < self.N:
            self.missing_images = set([i+1 for i in range(seqs[-1]) if i+1 not in seqs])
        dcim                  = dcmread(join(dirpath,filenames[0]))
        self.image_plane      = self.get_image_plane(dcim.ImageOrientationPatient)
        self.description      = dcim.SeriesDescription
        self.patient_position = dcim.PatientPosition

    def images(self):
        for i in range(1,len(self)+1):
            if i not in self.missing_images:
                yield  dcmread(join(self.dirpath,f'Image-{i}.dcm'))

    # get_image_plane
    #
    # Snarfed from https://www.kaggle.com/davidbroberts/determining-mr-image-planes
    def get_image_plane(self,loc):
        row_x = round(loc[0])
        row_y = round(loc[1])
        row_z = round(loc[2])
        col_x = round(loc[3])
        col_y = round(loc[4])
        col_z = round(loc[5])

        if row_x == 1 and row_y == 0 and col_x == 0 and col_y == 0:  return "Coronal"

        if row_x == 0 and row_y == 1 and col_x == 0 and col_y == 0:  return "Sagittal"

        if row_x == 1 and row_y == 0 and col_x == 0 and col_y == 1:  return "Axial"

        return "Unknown"

    def __len__(self):
        return self.N

    def image_files(self):
        for i in range(1,self.N+1):
            if i not in self.missing_images:
                yield join(self.dirpath,f'Image-{i}.dcm')



class MRI_Study:
    def __init__(self,name,path):
        self.series = None
        for dirpath, dirnames, filenames in walk(path):
            if self.series == None:
                self.series = {series_name: MRI_Series(series_name) for series_name in dirnames}
            else:
                path_components = normpath(dirpath).split(sep)
                series = self.series[path_components[-1]]
                series.add_images(dirpath,filenames)
                if len(series.missing_images)>0:
                    warn(f'Study {name} {series.name} has images missing: {series.missing_images}')

    def get_series(self):
        for name in ['FLAIR', 'T1w', 'T1wCE', 'T2w']:
            yield self.series[name]

class MRI_Dataset:
    def __init__(self,path,folder):
        self.studies = {name:MRI_Study(name,join(path,folder,name)) for name in listdir(join(path,folder))}

class Labelled_MRI_Dataset(MRI_Dataset):
    def __init__(self,path,folder,labels='train_labels.csv'):
        super().__init__(path,folder)
        self.labels = read_csv(join(path,labels),dtype={'BraTS21ID':str})

def plot_orbit(study):
    fig       = figure(figsize=(20,20))
    ax        = axes(projection='3d')

    for series in study.get_series():
        xs = []
        ys = []
        zs = []
        s  = []
        for dcim in series.images():
            xs.append(dcim.ImagePositionPatient[0] )
            ys.append(dcim.ImagePositionPatient[1] )
            zs.append(dcim.ImagePositionPatient[2] )
            s.append(10 if dcim.pixel_array.sum()> 0 else 1)
        ax.scatter(xs,ys,zs,
                   label = f'{dcim.SeriesDescription}: {dcim.PatientPosition} {series.get_image_plane(dcim.ImageOrientationPatient)}',
                   s     = s)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    title(dcim.PatientID)
    ax.legend()
    savefig(f'{dcim.PatientID}')

if __name__=='__main__':
    training = Labelled_MRI_Dataset(r'D:\data\rsna','train')
    test     = MRI_Dataset(r'D:\data\rsna','test')
    plot_orbit(training.studies['00000'])
    show()
