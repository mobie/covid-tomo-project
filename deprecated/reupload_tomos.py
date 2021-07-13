import os
import json
from subprocess import run

ROOT = 'data'


def get_mc_command(dataset, volume):
    bucket_name = 'covid-tomography'
    data_path = os.path.join(ROOT, dataset, 'images', 'local', volume)
    assert os.path.exists(data_path), data_path
    path_in_bucket = os.path.relpath(data_path, start=ROOT)
    full_s3_path = f'embl/{bucket_name}/{path_in_bucket}'
    mc_command = f"mc cp -r {os.path.relpath(data_path)}/ {full_s3_path}/"
    return mc_command


def upload_all_segmentations():
    with open('./data/datasets.json') as f:
        datasets = json.load(f)['datasets']

    volume = 'em-tomogram-segmentation.n5'
    for ds in datasets:
        mc_command = get_mc_command(ds, volume)
        print("Running", mc_command)
        mc_command = mc_command.split()
        run(mc_command)


# updates done:
# Calu_MOI5_6h_K2 - rawdata
# Calu3_MOI5_12h_E3 - rawdata
# Calu3_MOI0.5_24h_H2 - rawdata
# E2094_mock_O1 - rawdata
# Calu3_MOI5_24h_C2 - rawdata
if __name__ == '__main__':
    command = get_mc_command('Calu3_MOI0.5_24h_H2', 'em-tomogram-segmentation.n5')
    print(command)
    # upload_all_segmentations()
