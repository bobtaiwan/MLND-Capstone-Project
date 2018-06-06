
import pathlib
from tflearn.data_utils import build_hdf5_image_dataset
import h5py
from tqdm import tqdm

def get_label(windows_path):
    fileName = (str(windows_path)).split('\\')[-1]
    label = int(fileName[0:fileName.find('.')].split('_')[1])-1
    return label


if __name__=='__main__':
    for name in ['train', 'val']:
        print('{0} h5 file start to create'.format(name))
        directory = pathlib.Path('data/'+name)
        h5plan_name = pathlib.Path('h5plan.txt')
        output_filename = pathlib.Path(name+'_128.h5')
        with open(h5plan_name, "w") as text_file:
            for windows_path in tqdm(list(directory.glob('*.jpg'))):
                text_file.write('{0} {1}\n'.format(str(windows_path), get_label(windows_path)))

        build_hdf5_image_dataset(h5plan_name, image_shape=(128, 128), mode='file', output_path=output_filename,
                            categorical_labels=True, normalize=True)

        print('{0} h5 file created'.format(name))
        train_data = h5py.File(output_filename, 'r')
        X_train, Y_train = train_data['X'], train_data['Y']
        print(Y_train[:])