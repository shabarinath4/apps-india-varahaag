import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.utils import utilities
from tqdm import tqdm
import sys

sys.path.append('C:\\Users\\shaba\\Desktop\\1.1 Crop ID - Domain Adaptation\\presto\\presto-main\\openmapflow-main\\openmapflow')
sys.path.append('C:\\Users\\shaba\\Desktop\\1.1 Crop ID - Domain Adaptation\\presto\\presto-main\\openmapflow-main')
sys.path.append('C:\\Users\\shaba\\Desktop\\1.1 Crop ID - Domain Adaptation\\algos-crop_identification-deCIFR\\presto')
sys.path.append('/gpfs/data1/cmongp2/decifr/presto/presto-main/openmapflow-main/openmapflow')
sys.path.append('/gpfs/data1/cmongp2/decifr/presto/presto-main/openmapflow-main')
sys.path.append('/gpfs/data1/cmongp2/decifr/presto')

import presto

class PreTrainerPresto(object):
      def __init__(self, df, TOTAL_TS, LAT_LON_COLNAME, MONTH, features, TARGET):
        self.df = df 
        self.TOTAL_TS = TOTAL_TS
        self.LAT_LON_COLNAME = LAT_LON_COLNAME
        self.MONTH = MONTH
        self.features = features
        self.TARGET = TARGET
        S1_PRESTO_VALID_BANDS = ['VV', 'VH']
        S2_PRESTO_VALID_BANDS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12']
        S1_EXISTING_BANDS = list(set([f.split('_')[1] for f in features if f.split('_')[1] in  S1_PRESTO_VALID_BANDS]))
        S2_EXISTING_BANDS = list(set([f.split('_')[1] for f in features if f.split('_')[1] in  S2_PRESTO_VALID_BANDS]))
        self.S1_BANDS_SELECTED = [f for f in S1_EXISTING_BANDS if f in S1_PRESTO_VALID_BANDS]
        self.S2_BANDS_SELECTED = [f for f in S2_EXISTING_BANDS if f in S2_PRESTO_VALID_BANDS]
 
      def construct_presto_data(self):
        df_presto_S1 = pd.DataFrame([], columns=self.S1_BANDS_SELECTED)
        df_presto_S2 = pd.DataFrame([], columns=self.S2_BANDS_SELECTED)
        geometry = utilities.Geometry()
        df = geometry.get_lat_lon(self.df, self.LAT_LON_COLNAME)

        presto_s1 = np.empty((len(df), len(self.TOTAL_TS), len(self.S1_BANDS_SELECTED)))
        for row in range(len(df)):
            for time in range(len(self.TOTAL_TS)):
                for band in range(len(self.S1_BANDS_SELECTED)):
                    presto_s1[row, time, band] = df.loc[row, str(self.TOTAL_TS[time]) + '_' + str(self.S1_BANDS_SELECTED[band])]

        presto_s2 = np.empty((len(df), len(self.TOTAL_TS), len(self.S2_BANDS_SELECTED)))
        for row in range(len(df)):
            for time in range(len(self.TOTAL_TS)):
                for band in range(len(self.S2_BANDS_SELECTED)):
                    presto_s2[row, time, band] = df.loc[row, str(self.TOTAL_TS[time]) + '_' + str(self.S2_BANDS_SELECTED[band])]

        return presto_s1, presto_s2, self.S1_BANDS_SELECTED, self.S2_BANDS_SELECTED, df[['latitude', 'longitude']]
      
      def get_presto_features(self):

        batch_size = 64
        pretrained_model = presto.Presto.load_pretrained()
        pretrained_model.eval()
      
        arrays, masks, latlons, image_names, labels, dynamic_worlds = [], [], [], [], [], []
        presto_S1, presto_S2, self.S1_BANDS_SELECTED, self.S2_BANDS_SELECTED, LAT_LON = self.construct_presto_data()
        LAT_LON = torch.tensor(np.array(LAT_LON))
        print('presto_S1', presto_S1.shape)
        print('presto_S2', presto_S2.shape)
        presto_S1 = torch.from_numpy(presto_S1).float()
        presto_S2 = torch.from_numpy(presto_S2).float()
        y = self.df[self.TARGET]

        for p in range(len(presto_S1)):
            x, mask, dynamic_world = presto.construct_single_presto_input(
                s1=presto_S1[p, :, :] if self.S1_BANDS_SELECTED else None, s1_bands=self.S1_BANDS_SELECTED if self.S1_BANDS_SELECTED else None,
                s2=presto_S2[p, :, :], s2_bands=self.S2_BANDS_SELECTED)
            arrays.append(x)
            masks.append(mask)
            dynamic_worlds.append(dynamic_world)
            labels.append(y[p])
            image_names.append(y[p])

        presto_data = (torch.stack(arrays, axis=0),
                        torch.stack(masks, axis=0),
                        torch.stack(dynamic_worlds, axis=0),
                        LAT_LON,
                        torch.tensor(labels),
                        image_names)

        month = torch.tensor([0] * presto_data[0].shape[0]).long()

        dl = DataLoader(
            TensorDataset(
                presto_data[0].float(),  # x
                presto_data[1].bool(),  # mask
                presto_data[2].long(),  # dynamic world
                presto_data[3].float(),  # latlons
                month
            ),
            batch_size=batch_size,
            shuffle=False,
        )
        features_list = []
        for (x, mask, dw, latlons, month) in tqdm(dl):
            with torch.no_grad():
                encodings = (
                    pretrained_model.encoder(
                        x, dynamic_world=dw, mask=mask, latlons=latlons, month=month
                    )
                    .cpu()
                    .numpy()
                )
                features_list.append(encodings)
        features_np = np.concatenate(features_list)

        X, y = features_np, presto_data[4].numpy()

        return X, y