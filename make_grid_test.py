import os
import json
from glob import glob

import pandas as pd

from mobie.initialization import make_dataset_folders
from mobie.import_data.util import downscale
from mobie.metadata import add_dataset

IN_FOLDER = '/g/emcf/common/5792_Sars-Cov-2/Exp_300420/TEM/Tomography/raw_data/E2094_mock_O1/bdv/tomos'


def create_grid_metadata(dataset_folder, files):
    im_dict_path = os.path.join(dataset_folder, 'images', 'images.json')
    if os.path.exists(im_dict_path):
        with open(im_dict_path) as f:
            im_dict = json.load(f)
    else:
        im_dict = {}

    local_files = []
    remote_files = [lf.replace('local', 'remote') for lf in local_files]

    im_dict.update({
        "grid-test-dataset": {
            "type": "grid",
            "gridSize": [2, 2],
            "storage": {
                "local": local_files,
                "remote": remote_files
            },
            "initialGridPositions": [
                [0, 0],
                [0, 1],
                [1, 0],
                [1, 1]
            ],
            "color": "white",
            "contrastLimits": [
                0.0,
                255.0
            ],
            "tableFolder": "tables/grid-test-dataset"
        }
    })

    with open(im_dict_path, 'w') as f:
        json.dump(im_dict, f, sort_keys=True, indent=2)


def make_grid_test():
    root_out = './data'
    dataset_name = 'grid-test-dataset'
    dataset_folder = make_dataset_folders(root_out, dataset_name)

    chunks = (32, 128, 128)
    resolution = 3 * [1.558]
    scale_factors = [[1, 2, 2], [1, 2, 2], [1, 2, 2], [2, 2, 2]]

    pattern = os.path.join(IN_FOLDER, '*.h5')
    files = glob(pattern)
    files.sort()
    files = files[:4]

    xml_paths = []
    name_tamplate = 'em-tomogram-%i'

    for ii, file_ in enumerate(files, 1):
        out_key = 'setup0/timepoint0/s0'
        data_name = name_tamplate % ii
        tmp_folder = f'tmp_grid-test_{data_name}'

        data_path = os.path.join(dataset_folder, 'images', 'local', f'{data_name}.n5')
        xml_path = os.path.join(dataset_folder, 'images', 'local', f'{data_name}.xml')

        downscale(data_path, out_key, data_path, resolution, scale_factors, chunks,
                  tmp_folder=tmp_folder, target='local', max_jobs=8,
                  block_shape=chunks, library='skimage')
        xml_paths.append(xml_path)

    create_grid_metadata(dataset_folder, xml_paths)
    add_dataset(root_out, dataset_name, is_default=True)


def make_image_table():
    table_folder = "tables/grid-test-dataset"
    os.makedirs(table_folder, exist_ok=True)
    table_path = os.path.join(table_folder, 'default.csv')

    values = [
        [0, "tomo1"],
        [1, "tomo2"],
        [2, "tomo3"],
        [3, "tomo4"]
    ]

    columns = ['image_id', 'image_name']

    table = pd.DataFrame(values, columns=columns)
    table.to_csv(table_path, index=False, sep='\t')


if __name__ == '__main__':
    make_grid_test()
    make_image_table()
