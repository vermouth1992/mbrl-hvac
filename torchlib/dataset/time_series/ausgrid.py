"""
Functions to load and process Ausgrid dataset.
The dataset contains 300 users with their location, PV production and electrical consumption.
The timeline for this dataset is 3 years separated in 3 files.
"""

import os
import pickle

import numpy as np
import pandas as pd
from pandas.tseries.offsets import Day
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

from ...common import enable_cuda

DATA_PATH_ROOT = os.path.expanduser('~/Documents/Deep_Learning_Resources/datasets/ausgrid')

FILE_PATH_DICT = {
    '2010-2011': '2010-2011 Solar home electricity data.csv',
    '2011-2012': '2011-2012 Solar home electricity data.csv',
    '2012-2013': '2012-2013 Solar home electricity data.csv'
}

DATA_FRAME_PATH_DICT = {
    '2011-2012': '2011-2012 Solar home electricity data.pkl'
}


def process_reshape_data_frame(year='2011-2012'):
    assert year in FILE_PATH_DICT
    fname = os.path.join(DATA_PATH_ROOT, FILE_PATH_DICT[year])
    d_raw = pd.read_csv(fname, skiprows=1, parse_dates=['date'], dayfirst=True, na_filter=False,
                        dtype={'Row Quality': str})
    d0, d1 = d_raw.date.min(), d_raw.date.max()
    index = pd.date_range(d0, d1 + Day(1), freq='30T', closed='left')
    customers = sorted(d_raw.Customer.unique())
    channels = ['GC', 'GG', 'CL']
    empty_cols = pd.MultiIndex(levels=[customers, channels], labels=[[], []], names=['Customer', 'Channel'])
    df = pd.DataFrame(index=index, columns=empty_cols)
    missing_records = []
    for c in customers:
        d_c = d_raw[d_raw.Customer == c]
        for ch in channels:
            d_c_ch = d_c[d_c['Consumption Category'] == ch]
            ts = d_c_ch.iloc[:, 5:-1].values.ravel()
            if len(ts) != len(index):
                missing_records.append((c, ch, len(ts)))
            else:
                df[c, ch] = ts

    d_customer_cap = d_raw[['Customer', 'Generator Capacity']]
    gen_cap = d_customer_cap.groupby('Customer')['Generator Capacity'].mean()

    d_customer_post = d_raw[['Customer', 'Postcode']]
    postcode = d_customer_post.groupby('Customer')['Postcode'].mean()

    return df, missing_records, gen_cap, postcode


def save_data_frame(year='2011-2012'):
    path = os.path.join(DATA_PATH_ROOT, DATA_FRAME_PATH_DICT[year])
    df, missing_records, gen_cap, postcode = process_reshape_data_frame(year)
    data_dict = {
        'df': df,
        'miss_records': missing_records,
        'gen_cap': gen_cap,
        'postcode': postcode
    }
    with open(path, 'wb') as f:
        pickle.dump(data_dict, f)


def load_data_frame(year='2011-2012'):
    path = os.path.join(DATA_PATH_ROOT, DATA_FRAME_PATH_DICT[year])
    with open(path, 'rb') as f:
        data_dict = pickle.load(f)
    df, missing_records, gen_cap, postcode = data_dict['df'], data_dict['miss_records'], data_dict['gen_cap'], \
                                             data_dict['postcode']
    return df, missing_records, gen_cap, postcode


def create_training_data(year='2011-2012'):
    """ Create numpy format of training data.
        We treat generation capacity and postcode as user profiles

    Args:
        year: aus year to extract

    Returns: data with shape (300, 366, 48, 2) -> (user, day, half hour, consumption/PV)
             generation capacity (300,)
             postcode (300,)
             user_id (300,)

    """
    df, _, gen_cap, postcode = load_data_frame(year)
    data = np.zeros(shape=(300, 366, 48, 2))
    for i in range(300):
        data[i] = df[i + 1].values.reshape(366, 48, -1)[:, :, :2]
    user_id = np.arange(0, 300)
    return data, gen_cap.values, postcode.values, user_id


class AusgridDataSet(Dataset):
    def __init__(self, year='2011-2012', train=True, transform=None, target_transform=None):
        data, gen_cap, postcode, user_id = create_training_data(year)
        if train:
            self.data = data[:, :300, :, :]
        else:
            self.data = data[:, 300:, :, :]
        self.num_days = self.data.shape[1]
        self.data = self.data.reshape((-1, 48, 2)).transpose((0, 2, 1))
        self.gen_cap = gen_cap
        self.postcode = postcode
        self.user_id = user_id
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        data = self.data[index]
        gen_cap = self.gen_cap[index // self.num_days]
        postcode = self.postcode[index // self.num_days]
        user_id = self.user_id[index // self.num_days]
        if self.transform is not None:
            data = self.transform(data)
        if self.target_transform is not None:
            gen_cap, postcode, user_id = self.target_transform(gen_cap, postcode, user_id)
        return data, (gen_cap, postcode, user_id)

    def __len__(self):
        return len(self.data)


def get_ausgrid_default_transform():
    return None


def get_ausgrid_dataset(train, transform=None):
    if transform is None:
        transform = get_ausgrid_default_transform()
    return AusgridDataSet(train=train, transform=transform)


def get_ausgrid_dataloader(train, batch_size=128, transform=None):
    kwargs = {'num_workers': 1, 'pin_memory': True} if enable_cuda else {}
    dataset = get_ausgrid_dataset(train=train, transform=transform)
    data_loader = DataLoader(dataset, batch_size, shuffle=True, **kwargs)
    return data_loader
