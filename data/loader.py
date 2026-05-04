import json
import os

import h5py
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from sklearn import preprocessing
from torch.utils.data import Dataset


torch.manual_seed(0)
np.random.seed(0)


class SentinelDataset(Dataset):
    def __init__(self, root_dir, json_file, is_train=True):
        self.fips_codes = []
        self.years = []
        self.file_paths = []

        data = json.load(open(json_file))

        if is_train is None:
            indices = range(len(data))
        else:
            np.random.seed(42)
            n_samples = len(data)
            perm = np.random.permutation(n_samples)
            split = int(0.9 * n_samples)
            indices = perm[:split] if is_train else perm[split:]

        for idx in indices:
            obj = data[idx]
            self.fips_codes.append(obj["FIPS"])
            self.years.append(obj["year"])
            tmp_path = []
            for relative_path in obj["data"]["sentinel"]:
                tmp_path.append(os.path.join(root_dir, relative_path))
            self.file_paths.append(tmp_path)

    def __len__(self):
        return len(self.fips_codes)

    def __getitem__(self, index):
        fips_code, year = self.fips_codes[index], self.years[index]
        file_paths = self.file_paths[index]
        temporal_list = []

        for file_path in file_paths:
            with h5py.File(file_path, "r") as hf:
                groups = hf[fips_code]
                for i, d in enumerate(groups.keys()):
                    if i % 2 == 0:
                        grids = np.asarray(groups[d]["data"])
                        temporal_list.append(torch.from_numpy(grids))

        x = torch.stack(temporal_list)
        return x, fips_code, year


class USDADataset(Dataset):
    def __init__(self, root_dir, json_file, crop_type="soybean", is_train=True):
        if crop_type == "cotton":
            self.select_cols = [
                "PRODUCTION, MEASURED IN 480 LB BALES",
                "YIELD, MEASURED IN LB / ACRE",
            ]
        else:
            self.select_cols = [
                "PRODUCTION, MEASURED IN BU",
                "YIELD, MEASURED IN BU / ACRE",
            ]

        self.fips_codes = []
        self.years = []
        self.state_ansi = []
        self.county_ansi = []
        self.file_paths = []

        data = json.load(open(json_file))

        if is_train is None:
            indices = range(len(data))
        else:
            np.random.seed(42)
            n_samples = len(data)
            perm = np.random.permutation(n_samples)
            split = int(0.9 * n_samples)
            indices = perm[:split] if is_train else perm[split:]

        for idx in indices:
            obj = data[idx]
            self.fips_codes.append(obj["FIPS"])
            self.years.append(obj["year"])
            self.state_ansi.append(obj["state_ansi"])
            self.county_ansi.append(obj["county_ansi"])
            self.file_paths.append(os.path.join(root_dir, obj["data"]["USDA"]))

    def __len__(self):
        return len(self.fips_codes)

    def __getitem__(self, index):
        state_ansi = self.state_ansi[index]
        county_ansi = self.county_ansi[index]

        df = pd.read_csv(self.file_paths[index])
        df["state_ansi"] = df["state_ansi"].astype(str).str.zfill(2)
        df["county_ansi"] = df["county_ansi"].astype(str).str.zfill(3)
        df = df[(df["state_ansi"] == state_ansi) & (df["county_ansi"] == county_ansi)]
        df = df[self.select_cols]

        x = torch.from_numpy(df.values).to(torch.float32)
        x = torch.log(torch.flatten(x, start_dim=0))
        return x, self.fips_codes[index], self.years[index]


class DataWrapper:
    def __init__(self, img_size=224, s=1, kernel_size=9, train=True):
        self.img_size = img_size
        self.s = s
        self.kernel_size = kernel_size
        self.transform = (
            self.get_simclr_pipeline_transform() if train else self.get_transform_val()
        )

    def __call__(self, x):
        x = x.to(torch.float32)
        xi = self.transform(x)
        xj = self.transform(x)
        return xi, xj

    def get_simclr_pipeline_transform(self):
        color_jitter = transforms.ColorJitter(
            0.8 * self.s, 0.8 * self.s, 0.8 * self.s, 0.2 * self.s
        )
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(size=self.img_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.GaussianBlur(kernel_size=self.kernel_size),
                transforms.Normalize([0.466, 0.471, 0.380], [0.195, 0.194, 0.192]),
            ]
        )

    def get_transform_val(self):
        return transforms.Compose(
            [
                transforms.CenterCrop(size=self.img_size),
                transforms.Normalize([0.466, 0.471, 0.380], [0.195, 0.194, 0.192]),
            ]
        )


class ScalarNorm:
    def __init__(self):
        self.norm = preprocessing.StandardScaler()

    def __call__(self, x, reverse=False):
        if not reverse:
            x = self.norm.fit_transform(x)
        else:
            x = self.norm.inverse_transform(x)
        return torch.from_numpy(x).to(torch.float32)
