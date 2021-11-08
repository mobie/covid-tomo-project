import mobie


def add_missing_bm(ds_folder):
    metadata = mobie.metadata.read_dataset_metadata(ds_folder)
    for name, source in metadata["sources"].items():
        source_type = list(source.keys())[0]
        source = source[source_type]
        menu_name = "tomo" if source_type == "image" else "position"
        if source_type == "image":
            view = mobie.metadata.get_default_view(source_type, name, menu_name=menu_name)
        else:
            view = mobie.metadata.get_default_view(source_type, name, menu_name=menu_name,
                                                   tables=["default.tsv"])
        mobie.metadata.add_view_to_dataset(ds_folder, name, view)


def add_all_missing():
    datasets = mobie.metadata.get_datasets("./data")
    for ds in datasets:
        add_missing_bm(f"./data/{ds}")


if __name__ == "__main__":
    add_all_missing()
