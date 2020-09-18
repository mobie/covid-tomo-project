import os
from glob import glob

import numpy as np
import h5py
import z5py

from elf.parallel import copy_dataset
from elf.transformation import matrix_to_parameters
from elf.transformation.affine import affine_matrix_3d

from mobie.initialization import make_dataset_folders
from mobie.metadata import add_to_image_dict, add_bookmark, add_dataset, have_dataset
from mobie.import_data.util import downscale

KEY = 't00000/s00/0/cells'
N_THREADS = 16


def get_plane_shape(tile_shape, grid, spacing):
    plane_shape = tuple(ts * n_points + space * (n_points - 1)
                        for ts, n_points, space in zip(tile_shape, grid, spacing))
    return plane_shape


def get_array_shape(files, chunk_size, volumes_per_row):
    shapes = []
    n_files = len(files)

    # assert n_files >= volumes_per_row, f"{n_files}, {volumes_per_row}"
    if n_files < volumes_per_row:
        volumes_per_row = n_files

    n_rows = n_files / float(volumes_per_row)
    if n_files % volumes_per_row == 0:
        n_rows = n_files // volumes_per_row
    else:
        n_rows = n_files // volumes_per_row + 1

    for file_path in files:
        with h5py.File(file_path, 'r') as f:
            shapes.append(np.array(list(f[KEY].shape))[None])

    shape_array = np.concatenate(shapes, axis=0)
    # make sure the shapes agree along the image plane axes
    assert np.all(shape_array[:, 1])
    assert np.all(shape_array[:, 2])

    tile_shape = tuple(shape_array[0, 1:])
    plane_shape = get_plane_shape(tile_shape, (n_rows, volumes_per_row), chunk_size)
    big_shape = (np.max(shape_array[:, 0]),) + plane_shape
    return big_shape, tile_shape


def get_center(grid_point,
               tile_shape, tile_chunks,
               shape, z_max):

    # find the lower corner in the plane
    lower_corner = tuple(ts * gp for ts, gp in zip(tile_shape, grid_point))
    lower_corner = tuple((lc + gp * tc) if gp > 0 else lc
                         for lc, gp, tc in zip(lower_corner, grid_point, tile_chunks))

    # center align the z-slices
    z0 = (z_max - shape[0]) // 2
    lower_corner = (z0,) + lower_corner

    center_point = tuple(lc + sh // 2 for lc, sh in zip(lower_corner, shape))

    return center_point


def write_grid(ds, files, tile_shape, volumes_per_row, dry_run=False):
    row_id = 0
    col_id = 0

    grid_to_centers = {}

    tile_chunks = ds.chunks[1:]
    z_max = ds.shape[0]
    for file_id, in_file in enumerate(files):

        with h5py.File(in_file, 'r') as f:
            ds_in = f[KEY]
            shape = ds_in.shape
            center_point = get_center((row_id, col_id),
                                      tile_shape, tile_chunks,
                                      shape, z_max)
            bounding_box = tuple(slice(ce - sh // 2, ce + sh // 2)
                                 for ce, sh in zip(center_point, shape))

            grid_to_centers[(row_id, col_id)] = center_point

            print("Copy file", file_id + 1, "/", len(files), "at grid position", row_id, col_id)
            if not dry_run:
                # FIXME not chunk aligned, so can only do this single threaded
                copy_dataset(ds_in, ds, roi_out=bounding_box,
                             verbose=True, n_threads=1)
                # verbose=True, n_threads=N_THREADS)

        col_id += 1
        if col_id % volumes_per_row == 0:
            col_id = 0
            row_id += 1

    return grid_to_centers


# think about adding this to the mobie tools
def make_grid_dataset(files, chunks,
                      output_path, output_key,
                      volumes_per_row=10, dry_run=False):
    big_shape, tile_shape = get_array_shape(files, chunks[1:], volumes_per_row)
    with h5py.File(files[0], 'r') as f:
        dtype = f[KEY].dtype

    with z5py.File(output_path, 'a') as f:
        ds = f.require_dataset(output_key, shape=big_shape, chunks=chunks, dtype=dtype,
                               compression='gzip')

    return write_grid(ds, files, tile_shape, volumes_per_row, dry_run=dry_run)


# From Paolo:
# Calu3_MOI5_24h_C2 have has pixel size 1.554 nm
# others have 1.558 nm
def get_resolution(dataset_name):
    if dataset_name == 'Calu3_MOI5_24h_C2':
        res_nm = 3 * [1.554]
    else:
        res_nm = 3 * [1.558]
    return [re / 1000. for re in res_nm]


def make_bookmarks(dataset_folder, grid_center_positions, files,
                   raw_name, resolution, overwrite=False):
    # add the default bookmark
    add_bookmark(dataset_folder, 'default', 'default',
                 overwrite=overwrite,
                 layer_settings={raw_name: {'contrastLimits': [0., 255.]}})

    # For now, the parameters for the offsets in the affine views are
    # derived from a linear fit to some views, see find_affines.py
    # there should be an analytical way to do this, need to discuss with Tischi ...
    # also it's weird that these are not quite symmetric in xy ...
    # linear fit parameter
    ax, bx = -0.91732143, -0.44715057475198905
    ay, by = -0.92478852, -0.4337483002173521

    ii = 0
    # add bookmarks for the grid positions
    for grid_pos, center in grid_center_positions.items():
        row_id, col_id = grid_pos

        fname = files[ii]
        bookmark_name = os.path.splitext(os.path.split(fname)[1])[0]
        print(bookmark_name)

        position = [ce * res for ce, res in zip(center, resolution)]

        # compute the correct view:
        # field of view that (roughly covers) one tomogram
        scale = 3 * [0.2745448396519001]

        # fixed z translation
        tz = 0.2745448396519001
        # translation in plane from linear fit to some bdv values ...
        tx = ax * row_id + bx
        ty = ay * col_id + by
        translation = [tz, ty, tx]

        view = affine_matrix_3d(scale=scale, translation=translation)
        view = matrix_to_parameters(view)

        add_bookmark(dataset_folder, 'default', bookmark_name,
                     position=position[::-1],
                     norm_view=view,
                     overwrite=overwrite)
        ii += 1


def create_mobie_dataset(dataset_name, root_in, is_default, volumes_per_row=10):

    root_out = './data'
    dataset_folder = make_dataset_folders(root_out, dataset_name)

    raw_name = 'em-tomogram'
    data_path = os.path.join(dataset_folder, 'images', 'local', f'{raw_name}.n5')
    xml_path = os.path.join(dataset_folder, 'images', 'local', f'{raw_name}.xml')

    chunks = (32, 128, 128)
    resolution = get_resolution(dataset_name)
    scale_factors = [[1, 2, 2], [1, 2, 2], [1, 2, 2], [2, 2, 2]]

    pattern = os.path.join(root_in, '*.h5')
    files = glob(pattern)
    files.sort()

    out_key = 'setup0/timepoint0/s0'
    grid_center_positions = make_grid_dataset(files, chunks, data_path, out_key,
                                              volumes_per_row=volumes_per_row, dry_run=False)

    tmp_folder = f'tmp_{dataset_name}'
    downscale(data_path, out_key, data_path, resolution, scale_factors, chunks,
              tmp_folder=tmp_folder, target='local', max_jobs=N_THREADS,
              block_shape=chunks, library='skimage')

    add_to_image_dict(dataset_folder, 'image', xml_path, add_remote=True)

    make_bookmarks(dataset_folder, grid_center_positions, files, raw_name, resolution)

    add_dataset(root_out, dataset_name, is_default)


def create_all_datasets():
    ds_names = [
        'Calu3_MOI0.5_24h_H2',
        'Calu3_MOI5_12h_E3',
        'Calu3_MOI5_24h_C2',
        'Calu_MOI5_6h_K2',
        'E2094_mock_O1'
    ]

    is_default = True
    for ds_name in ds_names:
        root = f'/g/emcf/common/5792_Sars-Cov-2/Exp_300420/TEM/Tomography/raw_data/{ds_name}/bdv/tomos'

        # skip datasets we have already addded
        if have_dataset('./data', ds_name):
            is_default = False
            continue

        create_mobie_dataset(ds_name, root, is_default=is_default)
        is_default = False


def update_bookmarks(dataset_name, root_in, volumes_per_row=10):

    root_out = './data'
    dataset_folder = os.path.join(root_out, dataset_name)

    raw_name = 'em-tomogram'
    data_path = os.path.join(dataset_folder, 'images', 'local', f'{raw_name}.n5')

    chunks = (32, 128, 128)
    resolution = get_resolution(dataset_name)

    pattern = os.path.join(root_in, '*.h5')
    files = glob(pattern)
    files.sort()

    out_key = 'setup0/timepoint0/s0'
    grid_center_positions = make_grid_dataset(files, chunks, data_path, out_key,
                                              volumes_per_row=volumes_per_row, dry_run=True)

    make_bookmarks(dataset_folder, grid_center_positions, files, raw_name, resolution, overwrite=True)


def create_test_dataset():
    root = './test_input'
    dataset_name = 'test'
    if have_dataset('./data', dataset_name):
        print("Updating bookmarks...")
        update_bookmarks(dataset_name, root, volumes_per_row=4)
    else:
        print("Creating test dataset ...")
        create_mobie_dataset(dataset_name, root, is_default=True, volumes_per_row=4)


def update_all_bookmarks():
    ds_names = [
        'Calu3_MOI0.5_24h_H2',
        'Calu3_MOI5_12h_E3',
        'Calu3_MOI5_24h_C2',
        'Calu_MOI5_6h_K2',
        # don't have write permissions, need to hack this one once the others work
        # 'E2094_mock_O1'
    ]

    for ds in ds_names:
        root_in = f'/g/emcf/common/5792_Sars-Cov-2/Exp_300420/TEM/Tomography/raw_data/{ds}/bdv/tomos'
        update_bookmarks(ds, root_in)


if __name__ == '__main__':
    update_bookmarks()
    # create_all_datasets()
    # create_test_dataset()
