import json
import os
from math import nan

import pandas as pd
import numpy as np

from mobie.metadata import add_to_image_dict
from mobie.tables import compute_default_table

from create_grid_datasets import get_resolution
from mock_segmentations import mock_segmentation

TABLE_NAME = './Revisions_Table S1_List of tomograms and annotations.xlsx'

DATASETS = {
    'Calu3_MOI0.5_24h_H2': 'T6 - 24h_MOI0.5',
    'Calu3_MOI5_12h_E3': 'T4 - 12h_MOI5',
    'Calu3_MOI5_24h_C2': 'T5 - 24h_MOI5',
    'Calu_MOI5_6h_K2': 'T3 - 6h_MOI5',
    'E2094_mock_O1': 'T7 - Mock'
}


def make_default_table(dataset, resolution):
    key = 'setup0/timepoint0/s0'
    tmp_folder = f'./tmp_seg_{dataset}'
    tmp_table = os.path.join(tmp_folder, 'default.csv')

    seg_path = os.path.join('data', dataset, 'images', 'local', 'em-tomogram-segmentation.n5')
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
    return default_table, label_table


def parse_table(table_path, dataset, valid_names):
    sheet_name = DATASETS[dataset]
    tomo_table = pd.read_excel(table_path, sheet_name=sheet_name, header=1)
    print(tomo_table.shape)
    cols = tomo_table.columns.values
    cols[0] = 'tomogram'
    tomo_table.columns = cols
    tomo_table = tomo_table[valid_names]
    tomo_table = tomo_table.replace(nan, 0)
    tomo_table = tomo_table.replace('x', 1)

    return tomo_table


def make_new_table(dataset, table_save_path, resolution, with_viral_structures):

    valid_names = [
        'tomogram'  # this is for the filename
    ]

    viral_names = [
        'DMVs',
        'Virions',
        'DMVs Opening',
        'DMS',
        'connectors',
        'Fused DMVs'
    ]
    cellular_names = [
        'zER',
        'Peroxisomes',
        'Golgi',
        'VTC',
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

    if with_viral_structures:
        valid_names = valid_names + viral_names + cellular_names
    else:
        valid_names = valid_names + cellular_names

    tomo_table = parse_table(TABLE_NAME, dataset, valid_names)
    # TODO need to handle mock cell dataset differently

    old_names = tomo_table['tomogram'].values.copy()
    new_names = ['%03i' % i for i in range(1, len(old_names) + 1)]
    tomo_table['tomogram'] = new_names

    name_dict = dict(zip(old_names, new_names))
    with open(f'tomo_names_{dataset}.json', 'w') as f:
        json.dump(name_dict, f)
    return
    default_table, label_table = make_default_table(dataset, resolution)

    table = np.concatenate([label_table.values,
                            tomo_table.values,
                            default_table.values],
                           axis=1)

    cols = np.concatenate([label_table.columns.values,
                           tomo_table.columns.values,
                           default_table.columns.values])
    table = pd.DataFrame(table, columns=cols)
    print(table.shape)
    table.to_csv(table_save_path, sep='\t', index=False)


def update_table(dataset):
    ds_folder = os.path.join('./data', dataset)
    seg_name = 'em-tomogram-segmentation'
    seg_path = os.path.join(ds_folder, 'images', 'local', seg_name + '.xml')

    scale_factor = (2, 8, 8)
    resolution = get_resolution(dataset)
    resolution = [res * sf for res, sf in zip(resolution, scale_factor)]

    table_folder = os.path.join(ds_folder, 'tables', seg_name)
    os.makedirs(table_folder, exist_ok=True)
    table_save_path = os.path.join(table_folder, 'default.csv')
    make_new_table(dataset, table_save_path, resolution,
                   with_viral_structures=dataset != 'E2094_mock_O1')

    if not os.path.exists(seg_path):
        tmp_folder = f'./tmp_seg_{dataset}'
        mock_segmentation(dataset, seg_name, scale_factor, resolution, tmp_folder)
        add_to_image_dict(ds_folder, 'segmentation', seg_path,
                          table_folder=table_folder)


if __name__ == '__main__':
    ds_names = list(DATASETS.keys())
    update_table(ds_names[1])
