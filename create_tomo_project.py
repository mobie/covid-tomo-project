import os
from glob import glob

import mobie
import numpy as np
import pandas as pd

ROOT = "/g/emcf/common/5792_Sars-Cov-2/Exp_300420/TEM/Tomography/raw_data/"

DS_NAMES = [
    "Calu3_MOI0.5_24h_H2",
    "Calu3_MOI5_12h_E3",
    "Calu3_MOI5_24h_C2",
    "Calu_MOI5_6h_K2",
    "E2094_mock_O1"
]

# TODO
DS_TO_SHEET = {
    "Calu3_MOI0.5_24h_H2": "T6 - 24h_MOI0.5",
    "Calu3_MOI5_12h_E3": "",
    "Calu3_MOI5_24h_C2": "",
    "Calu_MOI5_6h_K2": "",
    "E2094_mock_O1": ""
}

COL_NAMES = [
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


# From Paolo:
# Calu3_MOI5_24h_C2 have has pixel size 1.554 nm
# others have 1.558 nm
def get_resolution(dataset_name):
    if dataset_name == "Calu3_MOI5_24h_C2":
        res_nm = 3 * [1.554]
    else:
        res_nm = 3 * [1.558]
    return [re / 1000. for re in res_nm]


def to_tomo_name(tomo_path):
    name = os.path.splitext(os.path.split(tomo_path)[1])[0]
    if name.endswith("_hm"):
        name = name[:-3]
    return name


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
        tomo_name = to_tomo_name(tomo)
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

    grid_sources = [[to_tomo_name(tomo)] for tomo in tomos]
    view = mobie.metadata.bookmark_metadata.make_grid_view(
        ds_folder, view_name, grid_sources
    )

    ds_meta = mobie.metadata.read_dataset_metadata(ds_folder)
    ds_meta["views"]["default"] = view
    mobie.metadata.write_dataset_metadata(ds_folder, ds_meta)


def update_grid_table(ds_name):
    table_path = f"./data/{ds_name}/tables/default/default.tsv"
    grid_table = pd.read_csv(table_path, sep="\t")

    sheet_name = DS_TO_SHEET[ds_name]
    annotations = pd.read_excel("./annotation_table.xlsx", sheet_name=sheet_name, header=1)

    sources = grid_table["source"].values

    annotation_sources = annotations["Unnamed: 1"].values  # these are the file names
    # need to add an "_hm" to the name
    assert (np.unique(sources) == np.unique(annotation_sources)).all()

    # find the permutation from annotation_sources -> sources
    # source names are ordered, so we can take argsort here!
    permutation = np.argsort(annotation_sources)
    assert (sources == annotation_sources[permutation]).all()
    for col_name in COL_NAMES:
        if col_name not in annotations.columns:
            continue
        col = annotations[col_name].values[permutation]
        col[col == "x"] = 1
        col[col != 1] = 0
        grid_table[col_name] = col

    grid_table.to_csv(table_path, sep="\t", index=False)


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
    update_grid_table(ds_name)


# TODO need to do some more name shuffling and double check this with Martin
def create_project():
    for ds in DS_NAMES:
        create_dataset(ds)
        return
    mobie.validation.validate_project("./data")


if __name__ == '__main__':
    create_project()
