import numpy as np
import h5py
import os
import math
import torch
import torch.utils.data as data
import pdb
import time

from dataset.minmax_normalization import MinMaxNormalization
from dataset.data_fetcher import DataFetcher


class Dataset:
    datapath = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
    print('*' * 10 + 'DEBUG' + '*' * 10)
    print(datapath)

    def __init__(self, dconf, Ext_type, Train_mode, Data_type, Length, Is_seq, test_days=-1, datapath=datapath):
        self.dconf = dconf
        self.dataset = dconf.name
        self.len_close = dconf.len_close
        self.len_period = dconf.len_period
        self.len_trend = dconf.len_trend
        self.datapath = datapath
        self.ext_type = Ext_type
        self.train_mode = Train_mode
        self.data_type = Data_type
        self.length = Length
        self.is_seq = Is_seq

        if self.dataset == 'DENSITY':
            self.nb_flow = 1
            self.dim_h = 200
            self.dim_w = 200
            self.T = 24
            test_days = 3 if test_days == -1 else test_days

            self.m_factor = 1.

        self.len_test = test_days * self.T
        self.portion = dconf.portion

    def trainset_of(self, vec):
        return vec[:math.floor((len(vec) - self.len_test) * self.portion)]

    def testset_of(self, vec):
        return vec[-math.floor(self.len_test * self.portion):]

    def split(self, x, y, x_ave, x_ave_q, y_cls, y_typ):
        x_tra = self.trainset_of(x)
        x_tes = self.testset_of(x)

        x_ave_tra = self.trainset_of(x_ave)
        x_ave_tes = self.testset_of(x_ave)

        x_ave_q_tra = self.trainset_of(x_ave_q)
        x_ave_q_tes = self.testset_of(x_ave_q)

        y_tra = self.trainset_of(y)
        y_tes = self.testset_of(y)

        y_tra_cls = self.trainset_of(y_cls)
        y_tes_cls = self.testset_of(y_cls)

        y_tra_typ = self.trainset_of(y_typ)
        y_tes_typ = self.testset_of(y_typ)

        return x_tra, x_ave_tra, x_ave_q_tra, y_tra, y_tra_cls, y_tra_typ, x_tes, x_ave_tes, x_ave_q_tes, y_tes, y_tes_cls, y_tes_typ

    def load_data(self):
        """
        return value:
            X_train & X_test: [XC, XP, XT, Xext]
            Y_train & Y_test: vector
        """
        # read file and place all of the raw data in np.array. 'ts' means timestamp
        # without removing incomplete days

        # print(f'=============={self.inp_type} 输入加载成功！=============')
        raw_data = np.load('./data/DENSITY/raw_data/data.npy')
        raw_date = np.load('./data/DENSITY/raw_data/date.npy')

        if self.train_mode == 'train_split_2' and self.ext_type == 'ho_wd':
            inp_path = f'./data/DENSITY/AVG6_4_2/expectation_inp.npy'
            ext_cls_path = f'./data/DENSITY/AVG6_4_2/expectation_cls.npy'
            raw_data = raw_data[16*self.T:]
            raw_date = raw_date[16*self.T:]
        elif self.train_mode == 'train_split_1' and self.ext_type == 'ho_wd':
            inp_path = f'./data/DENSITY/AVG6_4_1/expectation_inp.npy'
            ext_cls_path = f'./data/DENSITY/AVG6_4_1/expectation_cls.npy'
        elif self.train_mode == 'train_split_1' and self.ext_type == 'ho':
            #deprecated
            inp_path = f'./data/DENSITY/AVG6_4_1/expectation_inp.npy'
            ext_cls_path = f'./data/DENSITY/AVG6_4_1/expectation_cls.npy'
        else:
            raise print('param error')

        all_average_data = np.load(inp_path, allow_pickle=True)
        new_average_data_list = list([all_average_data])
        all_ext_cls = np.load(ext_cls_path, allow_pickle=True)
        new_all_ext_cls = list([all_ext_cls])
        print('Preprocessing: Min max normalizing')
        raw_data = np.expand_dims(raw_data,axis=1)
        data_list = list([raw_data])
        ts_new_list = list([raw_date])

        mmn = MinMaxNormalization()
        train_dat = self.trainset_of(raw_data)
        mmn.fit(train_dat)
        new_data_list = [
            mmn.transform(data).astype('float32', copy=False)
            for data in data_list
        ]
        print('Context data min max normalizing processing finished!')

        x_list, y_list, x_ave_list, x_ave_q_list, y_typ_list, ts_x_list, ts_y_list = [], [], [], [], [], [], []
        for idx in range(len(ts_new_list)):
            x, x_ave, x_ave_q, y, y_typ, ts_x, ts_y = \
                DataFetcher(new_data_list[idx], ts_new_list[idx], new_average_data_list[idx], new_all_ext_cls[idx], self.T).fetch_data(self.dconf)
            x_list.append(x)
            y_list.append(y)

            x_ave_list.append(x_ave)
            x_ave_q_list.append(x_ave_q)
            y_typ_list.append(y_typ)

            ts_x_list.append(ts_x)  # list nest list nest list nest numpy.datetime64 class
            ts_y_list.append(ts_y)  # list nest list nest numpy.datetime64 class
        x_con = np.concatenate(x_list)
        y = np.concatenate(y_list)
        x_ave = np.concatenate(x_ave_list)
        x_ave_q = np.concatenate(x_ave_q_list)
        y_typ = np.concatenate(y_typ_list)
        ts_y = np.concatenate(ts_y_list)

        print(ts_y[0])
        print(ts_y[-self.len_test:])

        Y_Class = []
        for i in enumerate(ts_y[::self.T]):
            Y_Class.append(np.array(range(0, self.T)))
        y_cls = np.concatenate(Y_Class, axis=0).reshape(-1, 1)

        y_typ = y_typ.reshape(-1, 1)

        # (16464, 12, 32, 32) (16464, 2, 32, 32) (16464, 6) (16464,)
        x_con_tra, x_ave_tra, x_ave_q_tra, y_tra, y_cls_tra, y_typ_tra, x_con_tes, x_ave_tes, x_ave_q_tes, y_tes, y_cls_tes, y_typ_tes = self.split(
            x_con, y, x_ave, x_ave_q, y_cls, y_typ)

        # 是否使用多个序列长度求loss
        if self.is_seq:
            x_con_tra = x_con_tra[:-self.length+1]
            x_con_tes = x_con_tes[:-self.length+1]
            x_ave_tra = x_ave_tra[:-self.length+1]
            x_ave_tes = x_ave_tes[:-self.length+1]
            y_cls_tra = y_cls_tra[:-self.length+1]
            y_cls_tes = y_cls_tes[:-self.length+1]
            y_typ_tra = y_typ_tra[:-self.length+1]
            y_typ_tes = y_typ_tes[:-self.length+1]

            y_seq_tra = []
            for i, _ in enumerate(y_tra[:-self.length+1]):
                y_seq_tra.append(y_tra[i:i+self.length])
            y_seq_tra = np.stack(y_seq_tra)

            y_seq_tes = []
            for i, _ in enumerate(y_tes[:-self.length+1]):
                y_seq_tes.append(y_tes[i:i+self.length])
            y_seq_tes = np.stack(y_seq_tes)


        class TempClass:
            def __init__(self_2):
                self_2.X_con_tra = x_con_tra
                self_2.X_ave_tra = x_ave_tra
                self_2.X_ave_q_tra = x_ave_q_tra
                if self.is_seq:
                    self_2.Y_tra = y_seq_tra
                else:
                    self_2.Y_tra = y_tra
                self_2.Y_cls_tra = y_cls_tra
                self_2.Y_typ_tra = y_typ_tra

                self_2.X_con_tes = x_con_tes
                self_2.X_ave_tes = x_ave_tes
                self_2.X_ave_q_tes = x_ave_q_tes
                if self.is_seq:
                    self_2.Y_tes = y_seq_tes
                else:
                    self_2.Y_tes = y_tes
                self_2.Y_cls_tes = y_cls_tes
                self_2.Y_typ_tes = y_typ_tes

                self_2.img_mean = np.mean(train_dat, axis=0)
                self_2.img_std = np.std(train_dat, axis=0)
                self_2.mmn = mmn
                self_2.ts_Y_train = self.trainset_of(ts_y)
                self_2.ts_Y_test = self.testset_of(ts_y)

            def show(self_2):
                print(
                    "Run: X inputs shape: ", self_2.X_con_tra.shape, self_2.X_ave_tra.shape, self_2.X_ave_q_tra.shape,
                    self_2.X_con_tes.shape, self_2.X_ave_tes.shape, self_2.X_ave_q_tes.shape,
                    "Y inputs shape: ", self_2.Y_tra.shape, self_2.Y_cls_tra.shape, self_2.Y_typ_tra.shape,
                    self_2.Y_tes.shape, self_2.Y_cls_tes.shape, self_2.Y_typ_tes.shape,
                )
                print("Run: min~max: ", self_2.mmn.min, '~', self_2.mmn.max)

        return TempClass()


class TorchDataset(data.Dataset):
    def __init__(self, ds, mode='train'):
        super(TorchDataset, self).__init__()
        self.ds = ds
        self.mode = mode

    def __getitem__(self, index):
        if self.mode == 'train':
            X_con = torch.from_numpy(self.ds.X_con_tra[index])
            X_ave = torch.from_numpy(self.ds.X_ave_tra[index])
            X_ave_q = torch.from_numpy(self.ds.X_ave_q_tra[index])
            Y = torch.from_numpy(self.ds.Y_tra[index])
            Y_tim = torch.Tensor(self.ds.Y_cls_tra[index])
            Y_typ = torch.Tensor(self.ds.Y_typ_tra[index])
        else:
            X_con = torch.from_numpy(self.ds.X_con_tes[index])
            X_ave = torch.from_numpy(self.ds.X_ave_tes[index])
            X_ave_q = torch.from_numpy(self.ds.X_ave_q_tes[index])
            Y = torch.from_numpy(self.ds.Y_tes[index])
            Y_tim = torch.Tensor(self.ds.Y_cls_tes[index])
            Y_typ = torch.Tensor(self.ds.Y_typ_tes[index])

        return X_con.float(), X_ave.float(), X_ave_q.float(), Y.float(), Y_tim.float(), Y_typ.float()

    def __len__(self):
        if self.mode == 'train':
            return self.ds.X_con_tra.shape[0]
        else:
            return self.ds.X_con_tes.shape[0]


class DatasetFactory(object):
    def __init__(self, dconf, Ext_type, Train_mode, Data_type, Length, Is_seq):
        self.dataset = Dataset(dconf, Ext_type, Train_mode, Data_type, Length, Is_seq)
        self.ds = self.dataset.load_data()
        print('Show a list of dataset!')
        print(self.ds.show())

    def get_train_dataset(self):
        return TorchDataset(self.ds, 'train')

    def get_test_dataset(self):
        return TorchDataset(self.ds, 'test')


if __name__ == '__main__':
    class DataConfiguration:
        def __init__(self, Len_close, Len_period, Len_trend):
            super().__init__()

            # Data
            self.name = 'TaxiBJ'
            self.portion = 1.  # portion of data

            self.len_close = Len_close
            self.len_period = Len_period
            self.len_trend = Len_trend
            self.pad_forward_period = 0
            self.pad_back_period = 0
            self.pad_forward_trend = 0
            self.pad_back_trend = 0

            self.len_all_close = self.len_close * 1
            self.len_all_period = self.len_period * (1 + self.pad_back_period + self.pad_forward_period)
            self.len_all_trend = self.len_trend * (1 + self.pad_back_trend + self.pad_forward_trend)

            self.len_seq = self.len_all_close + self.len_all_period + self.len_all_trend
            self.cpt = [self.len_all_close, self.len_all_period, self.len_all_trend]

            self.interval_period = 1
            self.interval_trend = 7

            self.ext_flag = True
            self.ext_time_flag = True
            self.rm_incomplete_flag = True
            self.fourty_eight = True
            self.previous_meteorol = True

            self.dim_h = 200
            self.dim_w = 200

    df = DatasetFactory(DataConfiguration(4, 1, 1), Ext_type='ho_wd',Train_mode='train_split_2', Data_type='All', Length=6, Is_seq=0)
    ds = df.get_train_dataset()
    X, X_ave, X_ave_q, Y, Y_cls, Y_ext_cls = next(iter(ds))
    print('train:')
    print(X.size())
    print(X_ave.size())
    print(X_ave_q.size())
    print(Y.size())
    print(Y_cls)
    print(Y_ext_cls)

    # ds = df.get_train_dataset()
    # X, X_ave, X_ext, Y, Y_ext = next(iter(ds))
    # print('test:')
    # print(X.size())
    # print(X_ave.size())
    # print(X_ext.size())
    # print(Y.size())
    # print(Y_ext.size())
