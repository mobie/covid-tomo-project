import os
import json

DATASETS = [
    'Calu3_MOI0.5_24h_H2',
    'Calu3_MOI5_12h_E3',
    'Calu3_MOI5_24h_C2',
    'Calu_MOI5_6h_K2',
    'E2094_mock_O1'
]


def rename_bookmarks(dataset, drop_hm=False):
    with open(f"./tomo_names_{dataset}.json") as f:
        name_dict = json.load(f)

    name_dict = {os.path.splitext(k)[0]: v for k, v in name_dict.items()}

    bookmark_path = f'./data/{dataset}/misc/bookmarks/default.json'
    with open(bookmark_path) as f:
        bk = json.load(f)

    if drop_hm:
        bk = {k[:-3] if k != 'default' else k: v for k, v in bk.items()}

    file_names = set(bk.keys())
    table_names = set(name_dict.keys())
    print(file_names - table_names)
    print(table_names - file_names)

    new_bk = {}
    for name, val in bk.items():
        if name == 'default':
            new_bk[name] = val
        else:
            assert name in name_dict, name
            new_bk[name_dict[name]] = val

    with open(bookmark_path, 'w') as f:
        json.dump(new_bk, f, indent=2, sort_keys=True)


if __name__ == '__main__':
    rename_bookmarks(DATASETS[4], True)
