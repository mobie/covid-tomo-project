import os
from mobie.xml_utils import copy_xml_as_n5_s3

ROOT = 'data'


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


def ds_to_s3(ds_name):
    raw_name = 'em-tomogram'
    dataset_folder = os.path.join(ROOT, ds_name)
    assert os.path.exists(dataset_folder)

    out_path = os.path.join(dataset_folder, 'images', 'local', f'{raw_name}.n5')
    xml_path = os.path.splitext(out_path)[0] + '.xml'
    add_xml_for_s3(xml_path, out_path)


if __name__ == '__main__':
    ds_names = [
        'Calu3_MOI0.5_24h_H2',
        'Calu3_MOI5_12h_E3',
        'Calu3_MOI5_24h_C2',
        'Calu_MOI5_6h_K2'
    ]
    ds_to_s3(ds_names[2])
