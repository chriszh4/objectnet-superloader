import glob
import json

root = "/storage/dmayo2/datasets/objectnet/objectnet-1.0/images"
img_format = "png"

files = glob.glob(root+"/**/*."+img_format, recursive=True)



with open("mappings/folder_to_objectnet_label.json", "r") as f:
    folder_to_objectnet_label = json.load(f)

with open("mappings/objectnet_overlap_imagenet1k_idn.json", "r") as f:
    objectnet_overlap_imagenet1k_idn = json.load(f)

def map_to_imagenet(objectnet_folder_name):
    if objectnet_folder_name in objectnet_overlap_imagenet1k_idn:
        return objectnet_overlap_imagenet1k_idn[objectnet_folder_name]
    else:
        return ""

def data_from_path(path):
    return {"file_path" : path,
            "file_name" : path.split("/")[-1].split(".")[0],
            "objectnet_file_name" : path.split("/")[-2],
            "objectnet_label" : folder_to_objectnet_label[path.split("/")[-2]],
            "imagenet_id" : map_to_imagenet(path.split("/")[-2]),
            "subset" : "all" if map_to_imagenet(path.split("/")[-2]) == "" else "imagenet"}

data = [data_from_path(x) for x in files]

with open("objectnet_master.json", "w") as f:
    json.dump(data, f)

print(data[:5])