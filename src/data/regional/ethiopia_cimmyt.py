import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
from src.data.base.data_loader import DataLoader
from src.data.base.preprocess import Preprocess
import faiss
import time
import numpy as np
import pandas as pd
import rasterio as rio
from sklearn.preprocessing import LabelEncoder
import os
from shapely.geometry import Point
from tensorflow.keras.models import load_model

import sys
sys.path.append('C:\\Users\\shaba\\Desktop\\1.1 Crop ID - Domain Adaptation\\algos-crop_identification-deCIFR')

PATH_ID = 'ethiopia'
S1_TRAIN_DIR = None
S2_TRAIN_DIR = "data/train_ready/ethiopia_triplet_eos/croptype_cdse_l2a/train_ready"
TOTAL_TS = np.arange(0, 17, 1)
TS_SLICE_WIDTH = 17
TS_STEP = 1
TS_TO_PRED = []
TS_SLICES_TO_PRED = []
idx = 0
NORMALISE_STRATEGY = 'IQR'  # 'IQR','normalise'

YEARS = []
YEAR_COLNAME = 'Year'

CROPS_TO_DROP = 'Pigeon_Pea'
BINARISE_CROP = False
CROPS_TO_BINARISE = ['maize']
NUM_CLASSES = 2

UID = 'system:index'
UID_COLNAME = 'uid'

S1_BAND_NAME_MAP = {'VV': 'VV',
                    'VH': 'VH'}

S2_BAND_NAME_MAP = {'B1': 'B1',
                    'B2': 'B2',
                    'B3': 'B3',
                    'B4': 'B4',
                    'B5': 'B5',
                    'B6': 'B6',
                    'B7': 'B7',
                    'B8': 'B8',
                    'B8A': 'B8A',
                    'B9': 'B9',
                    'B10': 'B10',
                    'B11': 'B11',
                    'B12': 'B12'}

TARGET = 'TARGET'
TARGET_NAME_MAP = {'c_class': 'TARGET'}
GEOM_COLNAME = None
OUT_BANDS = 2  # currently writing first 2 predicted crops, their probabilities, and a confidence measure

S1_BANDS_SELECTED = []
S2_BANDS_SELECTED = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
S2_BANDS_IDX_SELECTED = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11]

S1_BANDS_TS = []
S1_BANDS_SELECTED_TS = []
for bn in S1_BAND_NAME_MAP.values():
    for ts in TOTAL_TS:
        S1_BANDS_TS.append(str(ts) + '_' + bn)
S1_BANDS_SELECTED_TS = [bn for bn in S1_BANDS_TS if bn.split('_')[1] in S1_BANDS_SELECTED]

S2_BANDS_TS = []
S2_BANDS_SELECTED_TS = []
for bn in S2_BAND_NAME_MAP.values():
    for ts in TOTAL_TS:
        S2_BANDS_TS.append(str(ts) + '_' + bn)
S2_BANDS_SELECTED_TS = [bn for bn in S2_BANDS_TS if bn.split('_')[1] in S2_BANDS_SELECTED]

S1_VIs = {}
S1_VIs_TS = []
for bn in S1_VIs.keys():
    for ts in TOTAL_TS:
        S1_VIs_TS.append(str(ts) + '_' + bn)

S2_VIs = {'NDVI':['B8', 'B4']}
S2_VIs_TS = []
for bn in S2_VIs.keys():
    for ts in TOTAL_TS:
        S2_VIs_TS.append(str(ts) + '_' + bn)

FEATURES = S1_BANDS_SELECTED + list(S1_VIs.keys()) + S2_BANDS_SELECTED + list(S2_VIs.keys())
FEATURES_TS = S1_BANDS_SELECTED_TS + S1_VIs_TS + S2_BANDS_SELECTED_TS + S2_VIs_TS

class EthiopiaCIMMYT():

    def __init__(self, RESULT_DIR, MODEL_DIR, MODE, log_run):
        self.RESULT_DIR = RESULT_DIR
        self.MODEL_DIR = MODEL_DIR
        self.MODE = MODE
        self.log_run = log_run

    def harmonise_CIMMYTEthiopia_S1_S2(self, df, feature_names_map, target_names_map, UID):

        columns_i = df.columns
        feature_names_actual = feature_names_map.keys()
        columns = []

        for c in columns_i:
            c_i = '_'.join(c.split('_')[1:])
            if c_i in feature_names_actual:
                columns = columns + [c]

        new_columns = [c.replace('_'.join(c.split('_')[1:]), feature_names_map['_'.join(c.split('_')[1:])]) for c in columns]
        df = df.rename(columns=dict(zip(columns, new_columns)))
        df = df.rename(columns=target_names_map)
        all_columns = new_columns + list(target_names_map.values()) + [UID]

        dataloader = DataLoader(TOTAL_TS=TOTAL_TS,
                                TS_SLICE_WIDTH=TS_SLICE_WIDTH,
                                TS_STEP=TS_STEP)
        # df, all_columns, feature_names_map = dataloader.make_VIs(df, S1_VIs, all_columns, feature_names_map)
        df, all_columns, feature_names_map = dataloader.make_VIs(df, S2_VIs, all_columns, feature_names_map)
        self.log_run.log_with_delta(f'Added Vegetation Indices {S2_VIs}')

        return df, all_columns, feature_names_map

    def load_cimmyt_ethiopia_S1_S2(self):

        # load train data
        if S1_TRAIN_DIR is not None:
            s1_files = os.listdir(S1_TRAIN_DIR)
            s1_files = [os.path.join(S1_TRAIN_DIR, f) for f in s1_files]
        else:
            s1_files = []

        if S2_TRAIN_DIR is not None:
            s2_files = os.listdir(S2_TRAIN_DIR)
            s2_files = [os.path.join(S2_TRAIN_DIR, f) for f in s2_files]
        else:
            s2_files = []

        s1_df = pd.DataFrame([])
        s2_df = pd.DataFrame([])

        for f in s1_files:
            s1_df_i = pd.read_csv(f)
            s1_df = pd.concat([s1_df, s1_df_i])

        for f in s2_files:
            s2_df_i = pd.read_csv(f)
            s2_df = pd.concat([s2_df, s2_df_i])

        df = s2_df
        # df = s1_df.merge(s2_df,left_on=GEOM_COLNAME,right_on=GEOM_COLNAME)
        # self.log_run.log_with_delta("Loaded train data")

        # select years
        try:
            df = df.loc[df[YEAR_COLNAME].isin(YEARS), :]
            self.log_run.log_with_delta(f"Selected Years {YEARS}")
        except:
            self.log_run.log_with_delta("No Year Column or no years selected by user")

        df, all_columns, S1_S2_BAND_NAME_MAP = self.harmonise_CIMMYTEthiopia_S1_S2(df, {**S1_BAND_NAME_MAP, **S2_BAND_NAME_MAP}, TARGET_NAME_MAP, UID)
        self.log_run.log_with_delta("Harmonised data")

        # drop crops
        try:
            df = df[df[TARGET] != CROPS_TO_DROP].reset_index(drop=True)
            self.log_run.log_with_delta(f"Dropped crops : {CROPS_TO_DROP}")
        except:
            self.log_run.log_with_delta(f"Crop is not present in the parent data : {CROPS_TO_DROP}")

        # df = df[df[TARGET].isin(['wheat', 'teff'])].reset_index(drop=True)
        df.loc[~df[TARGET].isin(['wheat', 'teff', 'barley']), TARGET] = 'othere'
        df.loc[df[TARGET].isin(['wheat', 'teff', 'barley']), TARGET] = 'cereals'
        # df = df[df[TARGET].isin(['wheat', 'teff']), TARGET].reset_index(drop=True)
        # drop crops
        try:
            if BINARISE_CROP:
                df.loc[~df[TARGET].isin(CROPS_TO_BINARISE), 'TARGET'] = 'Others'
                self.log_run.log_with_delta(f"Crop Binarise : {CROPS_TO_BINARISE}")
                NUM_CLASSES = 2
            else:
                NUM_CLASSES = len(df[TARGET].unique())
        except:
            self.log_run.log_with_delta(f"Crop is not present in the parent data : {CROPS_TO_BINARISE}")

        counts = df[TARGET].value_counts()
        df = df[df[TARGET].map(counts) > 10].reset_index(drop=True)
        NUM_CLASSES = len(df[TARGET].unique())

        # label encoding
        le = LabelEncoder()
        le.fit(df[TARGET])
        df[TARGET] = le.transform(df[TARGET])
        df[UID_COLNAME] = df[UID]
        classes = list(df[TARGET].unique())
        self.log_run.log_with_delta("Encoded Labels")

        features = S1_BANDS_SELECTED_TS + S2_BANDS_SELECTED_TS + S1_VIs_TS + S2_VIs_TS

        def get_band_ts(BAND_TS_I):
            rf = []
            for t in TOTAL_TS:
                rf = rf + [f for f in BAND_TS_I if f.split('_')[0] == str(t)]
            return rf

        raster_features = get_band_ts(S1_BANDS_TS)
        raster_features = raster_features + get_band_ts(S1_VIs_TS)
        raster_features = raster_features + get_band_ts(S2_BANDS_TS)
        raster_features = raster_features + get_band_ts(S2_VIs_TS)

        # preprocess data
        outlier_dict = {}
        normalise_dict = {}
        preprocess = Preprocess(df, features, TOTAL_TS, TARGET, outlier_dict, normalise_dict, self.MODE, self.RESULT_DIR, self.MODEL_DIR, self.log_run)
        df, outlier_dict, normalise_dict = preprocess.prep(S1_S2_BAND_NAME_MAP, NORMALISE_STRATEGY)
        self.log_run.log_with_delta("Data preprocessed")

        # split by plot id . Get plot medians
        select_columns = features + [TARGET] + [UID_COLNAME]
        if GEOM_COLNAME is not None:
            geom_df = df[[UID_COLNAME] + [GEOM_COLNAME]].drop_duplicates(subset=[UID_COLNAME])
        else:
            geom_df = df[[UID_COLNAME]].drop_duplicates(subset=[UID_COLNAME])
        median_df = df[select_columns].groupby(UID_COLNAME).median()
        median_df = median_df.dropna(subset=features + [TARGET], how='all')
        median_df = median_df.merge(geom_df, on=UID_COLNAME, how='left')
        X, y = np.array(median_df[features]), np.array(median_df[TARGET])
        self.log_run.log_with_delta("Got plot medians")

        return X, y, df, outlier_dict, normalise_dict, features, TARGET, NUM_CLASSES, S1_S2_BAND_NAME_MAP, OUT_BANDS, raster_features, le, TOTAL_TS

    def train_ready_data_gen(self, dir_path):
        x = np.load(os.path.join(dir_path, 'data.npy'))
        y = np.load(os.path.join(dir_path, 'crops.npy'))
        ids = np.load(os.path.join(dir_path, 'ids.npy'), allow_pickle=True)

        if os.path.exists(os.path.join(dir_path, 'latitudes.npy')):
            lat = np.load(os.path.join(dir_path, 'latitudes.npy'), allow_pickle=True) 
            lon = np.load(os.path.join(dir_path, 'longitudes.npy'), allow_pickle=True) 

        print(x)
        x = x[:, :, 0, 0, S2_BANDS_IDX_SELECTED]
        features = S1_BANDS_SELECTED_TS + S2_BANDS_SELECTED_TS  + S1_VIs_TS + S2_VIs_TS
        x = x.reshape(x.shape[0], x.shape[1]*x.shape[2], order='F')

        dataloader = DataLoader(TOTAL_TS=TOTAL_TS,
                        TS_SLICE_WIDTH=TS_SLICE_WIDTH,
                        TS_STEP=TS_STEP)
        x, features, feature_names_map = dataloader.make_VIs(pd.DataFrame(x, columns = S1_BANDS_SELECTED_TS + S2_BANDS_SELECTED_TS), {**S1_VIs, **S2_VIs}, S1_BANDS_SELECTED_TS + S2_BANDS_SELECTED_TS, {**S1_BAND_NAME_MAP, **S2_BAND_NAME_MAP})
        x = x.astype(float)
        x[x == 0] = np.nan
        df = pd.DataFrame(x, columns=features)
        self.log_run.log_with_delta(f"Data Loaded : {df.shape}")
        df[TARGET] = y
        df['system:index'] = ids

        def lat_lon_to_geojson(lat, lon):
            return {
                "type": "Point",
                "coordinates": [lon, lat]
            }
    
        if os.path.exists(os.path.join(dir_path, 'latitudes.npy')):
            df['lat'] = lat
            df['lon'] = lon
            df['.geo'] = df.apply(lambda row: lat_lon_to_geojson(row['lat'], row['lon']), axis=1)

        if not os.path.exists(os.path.join(dir_path, 'train_ready')):
            os.makedirs(os.path.join(dir_path, 'train_ready'))

        df.to_csv(os.path.join(dir_path, 'train_ready', 'train.csv'))
        exit()

    def deploy_CIMMYTEthiopia_S2_fsl(self,
                                     idx,
                                     MODEL_ID,
                                     PATH_ID,
                                     TARGET,
                                     MODE,
                                     BAND_NAME_MAP,
                                     OUT_BANDS,
                                     raster_features,
                                     features,
                                     outlier_dict,
                                     normalise_dict,
                                     INFERENCE_IN_DIR,
                                     INFERENCE_DIR,
                                     label_mapping,
                                     MODEL_DIR,
                                     rfc):

        time_start = time.time()
        infer = Infer()
        prototypes = np.load(MODEL_DIR + '/embeddings.npy')
        prototypes_labels = np.load(MODEL_DIR + '/embeddings_labels.npy')
        base_network = load_model(MODEL_DIR + '/model.keras')

        try:
            index = faiss.read_index(MODEL_DIR + '/index.faiss')
            labels = np.load(MODEL_DIR + '/labels.npy', allow_pickle=True)
        except:
            index = None
            labels = None

        log_folder = f"logs/{MODEL_ID}_inference_{idx[1]}"
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)

        log_run = LogRun(folder=log_folder, enable_logging=True)

        arr = np.load(idx[0])
        arr = arr.astype(float)
        print(arr.shape)
        print(arr[0,:,:,0])
        all_zero_pixels = np.all(arr == 0, axis=(0, 3))
        arr[arr == 0] = np.nan
        meta = np.load(idx[0].replace('datacube.npy', 'metadata.pickle.npy'), allow_pickle=True)[()]['geotiff_metadata']
        arr = arr[:, :, :, S2_BANDS_IDX_SELECTED]
        arr = arr[list(TOTAL_TS), :, :, :]
        arr_shp = arr.shape
        h, w, t, b = arr.shape[1], arr.shape[2], arr.shape[0], arr.shape[3]
        arr = arr.transpose(1, 2, 0, 3)
        arr = arr.reshape(h * w, t, b)
        arr = arr.reshape(h * w, t * b, order='F')

        time_dataloader = time.time()


        dataloader = DataLoader(TOTAL_TS=TOTAL_TS,
                                TS_SLICE_WIDTH=TS_SLICE_WIDTH,
                                TS_STEP=TS_STEP,
                                log_run=log_run)
        arr, features, feature_names_map = dataloader.make_VIs(pd.DataFrame(
            arr,
            columns = S1_BANDS_SELECTED_TS + S2_BANDS_SELECTED_TS), 
            {**S1_VIs, **S2_VIs}, S1_BANDS_SELECTED_TS + S2_BANDS_SELECTED_TS, 
            {**S1_BAND_NAME_MAP, **S2_BAND_NAME_MAP})
        arr_df = pd.DataFrame(arr, columns=features)
        arr_df.head().to_csv('head0.csv')
        log_run.log_with_delta(f"Data Loaded : {arr_df.shape}")

        time_preprocess = time.time()
        preprocess = Preprocess(
            arr_df,
            features,
            TOTAL_TS,
            TARGET,
            outlier_dict,
            normalise_dict,
            self.MODE,
            self.RESULT_DIR,
            self.MODEL_DIR,
            log_run)
        
        arr_df, outlier_dict, normalise_dict = preprocess.prep(
            BAND_NAME_MAP, 
            NORMALISE_STRATEGY)
        arr_df = arr_df[features]
        arr_df.head().to_csv('head1.csv')
        log_run.log_with_delta(f"Data preprocessed : {arr_shp}")

        time_inference = time.time()

        arr_df = np.expand_dims(np.array(arr_df), axis=-1)
        arr_pred, distances = infer.infer(
            base_network, 
            prototypes, 
            prototypes_labels, 
            index,
            labels,
            arr_df)
        log_run.log_with_delta("Model infered")

        time_postprocess = time.time()

        arr_pred = np.reshape(np.array(arr_pred),(1, arr_shp[1], arr_shp[2]))
        distances = np.reshape(np.array(distances),(1, arr_shp[1], arr_shp[2]))*10000
        out_rasster = np.concatenate([arr_pred, distances], axis=0)
        out_rasster[:, all_zero_pixels] = 256
        meta.update({"count":OUT_BANDS, "dtype":rio.int32, "nodata":256})
        # filename = '/'.join(idx[0].split('/')[-4:])
        # filedir = '/'.join(idx[0].split('/')[-4: -1])

        # if not os.path.exists(INFERENCE_DIR + '/' + filedir):
        #     os.makedirs(INFERENCE_DIR + '/' + filedir)
        # try:
        #     with rio.open(INFERENCE_DIR + '/' + filename.replace('datacube.npy', 'output.tif'), 'w', **meta) as outds:
        #         outds.write(out_rasster)
        #         print(f"{idx[1]}")
        #         log_run.log_with_delta(f"{idx[1]}")
        #         log_run.log_with_delta(f"Write successful : {INFERENCE_DIR + '/' + filename.replace('datacube.npy', 'output.tif')}")
        # except Exception as e:
        #     print(e)
        #     log_run.log_with_delta("WRITE FAILED ")
        #     pass

        # time_end = time.time()

        # print(time_preprocess - time_start)
        # print(time_inference - time_preprocess)
        # print(time_postprocess - time_inference)
        # print(time_end - time_postprocess)


        # filename = '/'.join(idx[0].split('/')[-4:])
        # filedir = '/'.join(idx[0].split('/')[-4: -1])

        # if not os.path.exists(INFERENCE_DIR + '/' + filedir):
        #     os.makedirs(INFERENCE_DIR + '/' + filedir)
        try:
            with rio.open(INFERENCE_DIR + '/' + 'output.tif', 'w', **meta) as outds:
                outds.write(out_rasster)
                print(f"{idx[1]}")
                log_run.log_with_delta(f"{idx[1]}")
                log_run.log_with_delta(f"Write successful : {INFERENCE_DIR  + '/' + 'output.tif'}")
        except Exception as e:
            print(e)
            log_run.log_with_delta("WRITE FAILED ")
            pass

        time_end = time.time()

        print(time_preprocess - time_start)
        print(time_inference - time_preprocess)
        print(time_postprocess - time_inference)
        print(time_end - time_postprocess)