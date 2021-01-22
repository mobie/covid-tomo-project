import os
import json
from glob import glob

import pandas as pd

from mobie.initialization import make_dataset_folders
from mobie.import_data.util import downscale
from mobie.metadata import add_dataset, add_to_image_dict
from mobie.xml_utils import copy_xml_as_n5_s3

ROOT = './data'
IN_FOLDER = '/g/emcf/common/5792_Sars-Cov-2/Exp_300420/TEM/Tomography/raw_data/E2094_mock_O1/bdv/tomos'


def create_grid_metadata(dataset_folder):
    im_dict_path = os.path.join(dataset_folder, 'images', 'images.json')
    with open(im_dict_path) as f:
        im_dict = json.load(f)

    images = ["em-tomogram-1", "em-tomogram-2", "em-tomogram-3", "em-tomogram-4"]
    im_dict.update({
        "collection-em-tomograms": {
            "images": images,
            "type": "collection",
            "layout": "autoGrid",
            "gridSize": [2, 2],
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


def add_xml_for_s3(xml_path, data_path):
    bucket_name = 'covid-tomography'
    xml_out_path = xml_path.replace('local', 'remote')
    if os.path.exists(xml_out_path):
        return
    path_in_bucket = os.path.relpath(data_path, start=ROOT)
    copy_xml_as_n5_s3(xml_path, xml_out_path,
                      service_endpoint='https://s3.embl.de',
                      bucket_name=bucket_name,
                      path_in_bucket=path_in_bucket,
                      authentication='Anonymous')

    print("In order to add the data to the EMBL S3, please run the following command:")
    full_s3_path = f'embl/{bucket_name}/{path_in_bucket}'
    mc_command = f"mc cp -r {os.path.relpath(data_path)}/ {full_s3_path}/"
    print(mc_command)


def make_grid_test_data():
    dataset_name = 'grid-test-dataset'
    dataset_folder = make_dataset_folders(ROOT, dataset_name)

    chunks = (32, 128, 128)
    resolution = 3 * [1.558]
    scale_factors = [[1, 2, 2], [1, 2, 2], [1, 2, 2], [2, 2, 2]]

    pattern = os.path.join(IN_FOLDER, '*.h5')
    files = glob(pattern)
    files.sort()
    files = files[:4]

    xml_paths = []
    name_template = 'em-tomogram-%i'

    in_key = 't00000/s00/0/cells'

    for ii, file_ in enumerate(files, 1):
        data_name = name_template % ii
        tmp_folder = f'tmp_grid-test_{data_name}'

        data_path = os.path.join(dataset_folder, 'images', 'local', f'{data_name}.n5')
        xml_path = os.path.join(dataset_folder, 'images', 'local', f'{data_name}.xml')

        downscale(file_, in_key, data_path, resolution, scale_factors, chunks,
                  tmp_folder=tmp_folder, target='local', max_jobs=8,
                  block_shape=chunks, library='skimage')
        add_xml_for_s3(xml_path, data_path)
        xml_paths.append(xml_path)

        add_to_image_dict(dataset_folder, 'image', xml_path)

    add_dataset(ROOT, dataset_name, is_default=True)


def make_image_table():
    dataset_folder = "./data/grid-test-dataset"
    table_folder = os.path.join(dataset_folder, "tables/grid-test-dataset")
    os.makedirs(table_folder, exist_ok=True)
    table_path = os.path.join(table_folder, 'default.csv')

    values = [
        ["em-tomogram-1", "tomo1"],
        ["em-tomogram-2", "tomo2"],
        ["em-tomogram-3", "tomo3"],
        ["em-tomogram-4", "tomo4"]
    ]

    columns = ['image_name', 'attribute']

    table = pd.DataFrame(values, columns=columns)
    table.to_csv(table_path, index=False, sep='\t')


if __name__ == '__main__':
    # make_grid_test_data()
    create_grid_metadata('./data/grid-test-dataset')
    make_image_table()
