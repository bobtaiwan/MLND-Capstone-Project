import pathlib
from tflearn.data_utils import build_hdf5_image_dataset
import h5py



def get_id(windows_path):
    fileName = (str(windows_path)).split('/')[-1]
    id = fileName[0:fileName.find('.')].split('_')[0]
    return id


if __name__=='__main__':
    directory = pathlib.Path('data/test')
    h5plan_name = pathlib.Path("test_h5plan.txt")
    #output_filename = directory.joinpath('train_112.h5')
    output_filename = pathlib.Path('test_128.h5')
    print('start proc')
    with open(h5plan_name, "w") as text_file:
        for windows_path in directory.glob('*.jpg'):
            text_file.write('{0} {1}\n'.format(str(windows_path), get_id(windows_path)))
    build_hdf5_image_dataset(h5plan_name, image_shape=(128, 128), mode='file', output_path=output_filename, categorical_labels=False, normalize=True)
    print('end proc')
    test_data = h5py.File(output_filename, 'r')
    X_train, ID = test_data['X'], test_data['Y']
    #print(Y for Y in Y_train[:] if Y==1)
    print(ID[:])