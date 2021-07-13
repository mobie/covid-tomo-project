import os
from math import nan
from glob import glob
from elf.io import open_file

from mobie.metadata import add_to_image_dict
from mobie.import_data.util import downscale
from mobie.tables import compute_default_table

import numpy as np
import pandas as pd
from create_grid_datasets import make_grid_dataset, get_resolution
# from mobie import


def mock_segmentation(ds_name, seg_name, scale_factor, resolution, tmp_folder):

    chunks = (32, 128, 128)

    root_in = f'/g/emcf/common/5792_Sars-Cov-2/Exp_300420/TEM/Tomography/raw_data/{ds_name}/bdv/tomos'
    pattern = os.path.join(root_in, '*.h5')
    files = glob(pattern)
    files.sort()

    raw_name = 'em-tomogram'
    dataset_folder = os.path.join('data', ds_name)
    data_path = os.path.join(dataset_folder, 'images', 'local', f'{raw_name}.n5')
    out_key = 'setup0/timepoint0/s0'

    volumes_per_row = min(10, len(files))
    grid_center_positions = make_grid_dataset(files, chunks, data_path, out_key,
                                              volumes_per_row=volumes_per_row, dry_run=True)

    with open_file(data_path, 'r') as f:
        ds = f[out_key]
        shape = ds.shape
    shape = tuple(sh // sf for sh, sf in zip(shape, scale_factor))

    seg_path = os.path.join(dataset_folder, 'images', 'local', f'{seg_name}.n5')
    f = open_file(seg_path, 'a')
    ds = f.require_dataset(out_key, shape=shape, compression='gzip', chunks=chunks,
                           dtype='uint16')
    # max_id = ds[:].max()
    # print(max_id)
    # ds.attrs['maxId'] = int(max_id)
    # return

    h5_key = 't00000/s00/0/cells'
    for label_id, (in_file, grid_center) in enumerate(zip(files, grid_center_positions.values()), 1):

        grid_center = [gc // sf for gc, sf in zip(grid_center, scale_factor)]

        with open_file(in_file, 'r') as f_in:
            vol_shape = f_in[h5_key].shape
        vol_shape = [vs // sf for vs, sf in zip(vol_shape, scale_factor)]
        bb = tuple(slice(gc - vs // 2, gc + vs // 2) for gc, vs in zip(grid_center, vol_shape))

        ds[bb] = label_id

    ds.attrs['maxId'] = label_id

    ds_factors = [[2, 2, 2]]
    downscale(seg_path, out_key, seg_path, resolution, ds_factors, chunks,
              tmp_folder=tmp_folder, target='local', max_jobs=4,
              block_shape=chunks, library='vigra', library_kwargs={'order': 0})


def compute_tomo_table(seg_path, table_path, tmp_folder, resolution, ds_name):
    key = 'setup0/timepoint0/s0'
    tmp_table = os.path.join(tmp_folder, 'default.csv')
    compute_default_table(seg_path, key, tmp_table, resolution,
                          tmp_folder=tmp_folder, target='local',
                          max_jobs=4)
    default_table = pd.read_csv(tmp_table, sep='\t')
    default_table = default_table.drop(axis=1, index=0)

    label_name = ["label_id"]
    valid_default_names = [
        "anchor_x",
        "anchor_y",
        "anchor_z",
        "bb_min_x",
        "bb_min_y",
        "bb_min_z",
        "bb_max_x",
        "bb_max_y",
        "bb_max_z"
    ]
    label_table = default_table[label_name]
    default_table = default_table[valid_default_names]
    print(label_table.shape)
    print(default_table.shape)

    tomo_table = './Table S1_List of tomograms and annotations.xlsx'
    sheet_name = '_'.join(ds_name.split('_')[1:3][::-1])

    valid_names = [
        'filename',  # this is for the filename
        'DMVs',
        'Virions',
        'DMS',
        'connectors',
        'zER',
        'Fused DMVs',
        'Peroxisomes',
        'Golgi',
        'ERGIC',
        'ER',
        'Mitochondria',
        'MVB/Lys',
        'Lamellar Bodies',
        'Lipid droplets',
        'autophagosomes',
        'Glycogen clusters',
        'Nucleus',
        'PM'
    ]

    tomo_table = pd.read_excel(tomo_table, sheet_name=sheet_name, header=1)
    print(tomo_table.shape)
    cols = tomo_table.columns.values
    cols[0] = 'filename'
    tomo_table.columns = cols
    tomo_table = tomo_table[valid_names]
    tomo_table = tomo_table.replace(nan, 0)
    tomo_table = tomo_table.replace('x', 1)

    # NOTE pd.concat doesn't work as expected (it adds another row)
    # so we concatenate by hand
    table = np.concatenate([label_table.values,
                            tomo_table.values,
                            default_table.values],
                           axis=1)

    cols = np.concatenate([label_table.columns.values,
                           tomo_table.columns.values,
                           default_table.columns.values])
    table = pd.DataFrame(table, columns=cols)
    print(table.shape)
    table.to_csv(table_path, sep='\t', index=False)


def create_grid_segmentation(ds_name, scale_factor=(2, 8, 8)):
    seg_name = 'em-tomogram-segmentation'
    dataset_folder = os.path.join('data', ds_name)

    resolution = get_resolution(ds_name)
    resolution = [res * sf for res, sf in zip(resolution, scale_factor)]

    tmp_folder = f'./tmp_seg_{ds_name}'
    mock_segmentation(ds_name, seg_name, scale_factor, resolution, tmp_folder)

    seg_path = os.path.join(dataset_folder, 'images', 'local', f'{seg_name}.n5')
    table_folder = os.path.join(dataset_folder, 'tables', seg_name)
    os.makedirs(table_folder, exist_ok=True)
    table_path = os.path.join(table_folder, 'default.csv')
    compute_tomo_table(seg_path, table_path, tmp_folder, resolution, ds_name)

    xml_path = os.path.join(dataset_folder, 'images', 'local', f'{seg_name}.xml')
    add_to_image_dict(dataset_folder, 'segmentation', xml_path,
                      table_folder=table_folder)


def check_table(ds_name):
    sheet_name = '_'.join(ds_name.split('_')[1:3][::-1])
    tomo_table = './Table S1_List of tomograms and annotations.xlsx'
    tomo_table = pd.read_excel(tomo_table, sheet_name=sheet_name, header=1)
    filenames = tomo_table.values[:, 0]
    filenames = [os.path.splitext(fname)[0] for fname in filenames]
    filenames = [ff[:-3] if ff.endswith('_hm') else ff for ff in filenames]
    filenames = set(filenames)

    root_in = f'/g/emcf/common/5792_Sars-Cov-2/Exp_300420/TEM/Tomography/raw_data/{ds_name}/bdv/tomos'
    pattern = os.path.join(root_in, '*.h5')
    files = glob(pattern)
    files.sort()
    files = [os.path.split(ff)[1] for ff in files]
    files = [os.path.splitext(ff)[0] for ff in files]
    files = [ff[:-3] if ff.endswith('_hm') else ff for ff in files]
    files = set(files)

    diff1 = files - filenames
    print(diff1)

    diff2 = filenames - files
    print(diff2)


def check_all(ds_names):
    for ds_name in ds_names:
        print(ds_name)
        check_table(ds_name)
        print()


if __name__ == '__main__':
    ds_names = [
        'Calu3_MOI0.5_24h_H2',
        'Calu3_MOI5_12h_E3',
        'Calu3_MOI5_24h_C2',
        'Calu_MOI5_6h_K2'
    ]
    check_all(ds_names)

    # create_grid_segmentation(ds_names[0])
    # create_grid_segmentation(ds_names[1])
    # create_grid_segmentation(ds_names[2])
    # create_grid_segmentation(ds_names[3])
