import json

path = "data/Spoken-ObjectNet-50k/metadata/"

with open(path+"SON-test.json", "r") as f:
    test_data = json.load(f)
with open(path+"SON-train.json", "r") as f:
    train_data = json.load(f)
with open(path+"SON-val.json", "r") as f:
    val_data = json.load(f)

data = [test_data, train_data, val_data]

metadata = {"audio_base_path": "/storage/chriszh/objectnet-superloader/data/Spoken-ObjectNet-50k/wavs/"}

for json_file in data:
    for d in json_file["data"]:
        metadata[d["image"]] = d
        del metadata[d["image"]]["image"]

with open("SON_master.json", "w") as f:
    json.dump(metadata, f)
