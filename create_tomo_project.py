import os
from glob import glob

import mobie
import pandas as pd

ROOT = '/g/emcf/common/5792_Sars-Cov-2/Exp_300420/TEM/Tomography/raw_data/'
DS_NAMES = [
    'Calu3_MOI0.5_24h_H2',
    'Calu3_MOI5_12h_E3',
    'Calu3_MOI5_24h_C2',
    'Calu_MOI5_6h_K2',
    'E2094_mock_O1'
]


# From Paolo:
# Calu3_MOI5_24h_C2 have has pixel size 1.554 nm
# others have 1.558 nm
def get_resolution(dataset_name):
    if dataset_name == 'Calu3_MOI5_24h_C2':
        res_nm = 3 * [1.554]
    else:
        res_nm = 3 * [1.558]
    return [re / 1000. for re in res_nm]


def add_tomos(tomos, ds_name):
    menu_name = "em-tomograms"
    tmp_root = f"tmp_folders/tmp_{ds_name}"
    os.makedirs(tmp_root, exist_ok=True)

    chunks = (32, 128, 128)
    resolution = get_resolution(ds_name)
    scale_factors = [[1, 2, 2], [1, 2, 2], [1, 2, 2], [2, 2, 2]]

    ds_folder = f"./data/{ds_name}"
    metadata = mobie.metadata.read_dataset_metadata(ds_folder)
    sources = metadata.get("sources", [])

    for tomo in tomos:
        tomo_name = os.path.splitext(os.path.split(tomo)[1])[0]
        if tomo_name in sources:
            continue
        tmp_folder = os.path.join(tmp_root, tomo_name)
        mobie.add_image(tomo, "t00000/s00/0/cells", "./data",
                        dataset_name=ds_name, image_name=tomo_name,
                        resolution=resolution, scale_factors=scale_factors,
                        chunks=chunks, menu_name=menu_name, tmp_folder=tmp_folder,
                        target="local", max_jobs=8)


def add_grid_view(tomos, ds_name):

    ds_folder = f"./data/{ds_name}"
    view_name = "default"

    view = mobie.metadata.bookmark_metadata.make_grid_view(
        ds_folder, view_name, tomos
    )

    ds_meta = mobie.metadata.read_dataset_metadata(ds_folder)
    ds_meta["views"]["default"] = view
    mobie.metadata.write_dataset_metadata(ds_folder, view)


def create_dataset(ds_name):
    tomos = glob(os.path.join(ROOT, ds_name, "bdv", "tomos", "*.h5"))
    n_tomos = len(tomos)
    if n_tomos == 0:
        tomos = glob(os.path.join(ROOT, ds_name, "bdv", "*.h5"))
        n_tomos = len(tomos)
    assert n_tomos > 0, ds_name
    tomos.sort()
    print("Add dataset", ds_name, "with", n_tomos, "tomograms")

    add_tomos(tomos, ds_name)
    add_grid_view(tomos, ds_name)


def create_project():
    for ds in DS_NAMES:
        create_dataset(ds)
        return


if __name__ == '__main__':
    create_project()
