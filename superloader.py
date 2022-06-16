from torchvision.datasets.vision import VisionDataset
from PIL import Image
import glob
import json
import os

import librosa
from utils import compute_spectrogram

class ObjectNetDataset(VisionDataset):
    """
    ObjectNet dataset.
    Args:
        root (string): Root directory where images are downloaded to. The images can be grouped in folders. (the folder structure will be ignored)
        dataset_json_file (string): Path to file that stores metadata for ObjectNet
        SON_json (string): Path to file that stores metadata for Spoken ObjectNet
        split (string): either "imagenet" which will return only objectnet images that are in imagenet,
            or "all" for all
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, 'transforms.ToTensor'
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        img_format (string): jpg
                             png - the original ObjectNet images are in png format
    """

    def __init__(self, root, dataset_json_file, SON_json, split="imagenet", audio_conf=None, transform=None, target_transform=None, transforms=None, img_format="jpg"):
        """Init ObjectNet pytorch dataloader."""
        super(ObjectNetDataset, self).__init__(root, transforms, transform, target_transform)

        self.loader = self.pil_loader
        self.img_format = img_format
        self.audio_conf = audio_conf if audio_conf else {}
        if split not in ["imagenet", "all"]:
            raise ValueError
        with open(dataset_json_file, "r") as f:
            dataset_json = json.load(f)
        self.imgs = []
        self.metadata = {} # maps image file path to other relevant metadata
        with open(SON_json, "r") as f:
            SON_metadata = json.load(f)
        for d in dataset_json:
            suffix = d["file_path"].split("/")[-2]+"/"+d["file_path"].split("/")[-1] #key for SON_json
            if not (d["subset"] == "all" and split == "imagenet"):
                self.imgs.append(d["file_path"])
                self.metadata[d["file_path"]] = {"objectnet_file_name" : d["objectnet_file_name"],
                                                 "objectnet_label" : d["objectnet_label"],
                                                 "imagenet_id" : d["imagenet_id"],
                                                 "wav" : SON_metadata["audio_base_path"]+SON_metadata[suffix]["wav"],
                                                 "text" : SON_metadata[suffix]["asr_text"],
                                                 "file" : suffix
                                                }
        for file in self.imgs:
            if not os.path.exists(file):
                raise FileNotFoundError

    def __getitem__(self, index):
        """
        Get an image, its label, audio, nframes, and caption text.
        audio is a FloatTensor of size (N_freq, N_frames) for spectrogram, or (N_frames) for waveform
        nframes is an integer

        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the image file name
        """
        img, target = self.getImage(index)
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        audio, nframes = self._LoadAudio(self.metadata[self.imgs[index]]["wav"])
        text = self.metadata[self.imgs[index]]["text"]
        return img, target, audio, nframes, text

    def getImage(self, index, preprocess = True):
        """
        Load the image and its relevant metadata.

        Args:
            index (int): Index
        Return:
            tuple: Tuple (image, target). target is the image file name
        """
        if preprocess:
            img = self.loader("data/objectnet-1.0/preprocessed_images/"+self.metadata[self.imgs[index]]["file"])
        else:
            img = self.loader(self.imgs[index])
            # crop out red border
            width, height = img.size
            cropArea = (2, 2, width-2, height-2)
            img = img.crop(cropArea)
        return (img, self.imgs[index]) #if want to return other metadata, use self.metadata[self.imgs[index]] 

    def __len__(self):
        """Get the number of ObjectNet images to load."""
        return len(self.imgs)

    def pil_loader(self, path):
        """Pil image loader."""
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
        
    def _LoadAudio(self, path):
        y, sr = librosa.load(path, None)
        logspec, n_frames = compute_spectrogram(y, sr, self.audio_conf)
        return logspec, n_frames


x = ObjectNetDataset(root = "/storage/dmayo2/datasets/objectnet/objectnet-1.0/images", dataset_json_file = "objectnet_master.json", SON_json = "SON_master.json", img_format = "png")
print(len(x))
print(x[0])
