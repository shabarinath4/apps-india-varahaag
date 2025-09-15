import numpy as np

class DataLoader(object):
    def __init__(self, band_names, no_data_value=np.nan):
        self.band_names = band_names
        self.no_data_value = no_data_value

    def NDVI(self, data, v):
        band_nir = v[0]
        band_red = v[1]

        nir = data[:, :, band_nir]
        red = data[:, :, band_red]

        mask = (nir == self.no_data_value) | (red == self.no_data_value)

        ndvi = (nir - red) / (nir + red + 1e-10)
        ndvi[mask] = self.no_data_value
        samples, time, bands = data.shape
        new_data = np.zeros((samples, time, bands + 1), dtype=np.float32)
        new_data[:, :, :bands] = data
        new_data[:, :, bands] = ndvi
        new_band_names = self.band_names + ['NDVI']

        return new_data, new_band_names

    def VHVV(self, data, v):
        band_vh = v[0]
        band_vv = v[1]

        vh = data[:, :, band_vh]
        vv = data[:, :, band_vv]

        mask = (vh == self.no_data_value) | (vv == self.no_data_value)

        vhvv = (vh - vv) / (vh + vv + 1e-10)
        vhvv[mask] = self.no_data_value

        samples, time, bands = data.shape
        new_data = np.zeros((samples, time, bands + 1), dtype=np.float32)
        new_data[:, :, :bands] = data
        new_data[:, :, bands] = vhvv

        new_band_names = self.band_names + ['VHVV']

        return new_data, new_band_names

    def CRI(self, data, v):
        band_green = v[0]
        band_blue = v[1]

        green = data[:, :, band_green]
        blue = data[:, :, band_blue]

        mask = (green == self.no_data_value) | (blue == self.no_data_value)

        cri = green / (blue + 1e-10)
        cri[mask] = self.no_data_value

        samples, time, bands = data.shape
        new_data = np.zeros((samples, time, bands + 1), dtype=np.float32)
        new_data[:, :, :bands] = data
        new_data[:, :, bands] = cri

        new_band_names = self.band_names + ['CRI']

        return new_data, new_band_names

    def CFI3(self, data, v, NDVI_TS):
        band_green = v[0]
        band_blue = v[1]

        green = data[:, :, band_green]
        blue = data[:, :, band_blue]

        mask = (green == self.no_data_value) | (blue == self.no_data_value)

        cri = green / (blue + 1e-10)
        cri[mask] = self.no_data_value

        samples, time, bands = data.shape
        new_data = np.zeros((samples, time, bands + 1), dtype=np.float32)
        new_data[:, :, :bands] = data
        new_data[:, :, bands] = cri

        new_band_names = self.band_names + ['CRI']

        return new_data, new_band_names
    
    def NDYI(self, data, v):
        band_green = v[0]
        band_blue = v[1]

        green = data[:, :, band_green]
        blue = data[:, :, band_blue]

        mask = (green == self.no_data_value) | (blue == self.no_data_value)

        ndyi = (green - blue) / (green + blue + 1e-10)
        ndyi[mask] = self.no_data_value

        samples, time, bands = data.shape
        new_data = np.zeros((samples, time, bands + 1), dtype=np.float32)
        new_data[:, :, :bands] = data
        new_data[:, :, bands] = ndyi

        new_band_names = self.band_names + ['NDYI']

        return new_data, new_band_names

    def BHDI1(self, data, v):
        band_re2 = v[0]
        band_blue = v[1]
        band_red = v[2]

        re2 = data[:, :, band_re2]
        blue = data[:, :, band_blue]
        red = data[:, :, band_red]

        mask = (re2 == self.no_data_value) | (blue == self.no_data_value) | (red == self.no_data_value)

        bhdi1 = re2 - blue + red
        bhdi1[mask] = self.no_data_value

        samples, time, bands = data.shape
        new_data = np.zeros((samples, time, bands + 1), dtype=np.float32)
        new_data[:, :, :bands] = data
        new_data[:, :, bands] = bhdi1

        new_band_names = self.band_names + ['BHDI1']

        return new_data, new_band_names

    def BHDI2(self, data, v):
        band_re2 = v[0]
        band_red = v[1]
        band_re1 = v[2]
        band_blue = v[3]

        re2 = data[:, :, band_re2]
        red = data[:, :, band_red]
        re1 = data[:, :, band_re1]
        blue = data[:, :, band_blue]

        mask = (re2 == self.no_data_value) | (red == self.no_data_value) | (re1 == self.no_data_value) | (blue == self.no_data_value)

        bhdi2 = re2 + red + re1 - blue
        bhdi2[mask] = self.no_data_value

        samples, time, bands = data.shape
        new_data = np.zeros((samples, time, bands + 1), dtype=np.float32)
        new_data[:, :, :bands] = data
        new_data[:, :, bands] = bhdi2

        new_band_names = self.band_names + ['BHDI2']

        return new_data, new_band_names

    def BHDI3(self, data, v):
        band_re1 = v[0]
        band_green = v[1]
        band_blue = v[2]
        band_red = v[3]

        re1 = data[:, :, band_re1]
        green = data[:, :, band_green]
        blue = data[:, :, band_blue]
        red = data[:, :, band_red]

        mask = (re1 == self.no_data_value) | (green == self.no_data_value) | (blue == self.no_data_value) | (red == self.no_data_value)

        bhdi3 = re1 + green - blue + red
        bhdi3[mask] = self.no_data_value

        samples, time, bands = data.shape
        new_data = np.zeros((samples, time, bands + 1), dtype=np.float32)
        new_data[:, :, :bands] = data
        new_data[:, :, bands] = bhdi3

        new_band_names = self.band_names + ['BHDI3']

        return new_data, new_band_names

    def BHDI4(self, data, v):
        band_green = v[0]
        band_blue = v[1]
        band_red = v[2]

        green = data[:, :, band_green]
        blue = data[:, :, band_blue]
        red = data[:, :, band_red]

        mask = (green == self.no_data_value) | (blue == self.no_data_value) | (red == self.no_data_value)

        bhdi4 = 2 * green - blue - red
        bhdi4[mask] = self.no_data_value

        samples, time, bands = data.shape
        new_data = np.zeros((samples, time, bands + 1), dtype=np.float32)
        new_data[:, :, :bands] = data
        new_data[:, :, bands] = bhdi4

        new_band_names = self.band_names + ['BHDI4']

        return new_data, new_band_names

    # def REP(self, data, v):
    #     band_re3 = v[0]
    #     band_red = v[1]
    #     band_re2 = v[2]
    #     band_re1 = v[3]

    #     re3 = data[:, :, band_re3]
    #     red = data[:, :, band_red]
    #     re2 = data[:, :, band_re2]
    #     re1 = data[:, :, band_re1]

    #     mask = (re3 == self.no_data_value) | (red == self.no_data_value) | (re2 == self.no_data_value) | (re1 == self.no_data_value)

    #     rep = 705 + (35 * (0.5 * (re3 + red) - re1)) / (re2 - re1 + 1e-10)
    #     rep[mask] = self.no_data_value

    #     samples, time, bands = data.shape
    #     new_data = np.zeros((samples, time, bands + 1), dtype=np.float32)
    #     new_data[:, :, :bands] = data
    #     new_data[:, :, bands] = rep

    #     new_band_names = self.band_names + ['REP']

    #     return new_data, new_band_names
    
    def NDRE(self, data, v):
        # v contains band indices: [red_edge_index, nir_index]
        band_re = v[0]
        band_nir = v[1]

        re = data[:, :, band_re]
        nir = data[:, :, band_nir]

        mask = (re == self.no_data_value) | (nir == self.no_data_value)

        ndre = (nir - re) / (nir + re + 1e-6)  # avoid divide-by-zero
        ndre[mask] = self.no_data_value

        samples, time, bands = data.shape
        new_data = np.zeros((samples, time, bands + 1), dtype=np.float32)
        new_data[:, :, :bands] = data
        new_data[:, :, bands] = ndre

        new_band_names = self.band_names + ['NDRE']

        return new_data, new_band_names

    def CIre(self, data, v):
        # v contains band indices: [red_edge_index, nir_index]
        band_re = v[0]
        band_nir = v[1]

        re = data[:, :, band_re]
        nir = data[:, :, band_nir]

        mask = (re == self.no_data_value) | (nir == self.no_data_value)

        cire = (nir / (re + 1e-6)) - 1  # avoid divide-by-zero
        cire[mask] = self.no_data_value

        samples, time, bands = data.shape
        new_data = np.zeros((samples, time, bands + 1), dtype=np.float32)
        new_data[:, :, :bands] = data
        new_data[:, :, bands] = cire

        new_band_names = self.band_names + ['CIre']

        return new_data, new_band_names
    
    def MTCI(self, data, v):
        # v contains band indices: [red_index, red_edge_index, nir_index]
        band_red = v[0]
        band_re = v[1]
        band_nir = v[2]

        red = data[:, :, band_red]
        re = data[:, :, band_re]
        nir = data[:, :, band_nir]

        mask = (red == self.no_data_value) | (re == self.no_data_value) | (nir == self.no_data_value)

        mtci = (nir - re) / (re - red + 1e-6)
        mtci[mask] = self.no_data_value

        samples, time, bands = data.shape
        new_data = np.zeros((samples, time, bands + 1), dtype=np.float32)
        new_data[:, :, :bands] = data
        new_data[:, :, bands] = mtci

        new_band_names = self.band_names + ['MTCI']
        return new_data, new_band_names

    def EVI2(self, data, v):
        # v contains band indices: [red_index, nir_index]
        band_red = v[0]
        band_nir = v[1]

        red = data[:, :, band_red]
        nir = data[:, :, band_nir]

        mask = (red == self.no_data_value) | (nir == self.no_data_value)

        evi2 = 2.5 * (nir - red) / (nir + 2.4 * red + 1.0)
        evi2[mask] = self.no_data_value

        samples, time, bands = data.shape
        new_data = np.zeros((samples, time, bands + 1), dtype=np.float32)
        new_data[:, :, :bands] = data
        new_data[:, :, bands] = evi2

        new_band_names = self.band_names + ['EVI2']
        return new_data, new_band_names

    def PRI_proxy(self, data, v):
        # v contains band indices: [green_index, green_i_index]
        band_green = v[0]
        band_green_i = v[1]

        green = data[:, :, band_green]
        green_i = data[:, :, band_green_i]

        mask = (green == self.no_data_value) | (green_i == self.no_data_value)

        pri = (green_i - green) / (green_i + green + 1e-6)
        pri[mask] = self.no_data_value

        samples, time, bands = data.shape
        new_data = np.zeros((samples, time, bands + 1), dtype=np.float32)
        new_data[:, :, :bands] = data
        new_data[:, :, bands] = pri

        new_band_names = self.band_names + ['PRI_proxy']
        return new_data, new_band_names

    def PSRI(self, data, v):
        # v contains band indices: [red_index, green_index, nir_index]
        band_red = v[0]
        band_green = v[1]
        band_nir = v[2]

        red = data[:, :, band_red]
        green = data[:, :, band_green]
        nir = data[:, :, band_nir]

        mask = (red == self.no_data_value) | (green == self.no_data_value) | (nir == self.no_data_value)

        psri = (red - green) / (nir + 1e-6)
        psri[mask] = self.no_data_value

        samples, time, bands = data.shape
        new_data = np.zeros((samples, time, bands + 1), dtype=np.float32)
        new_data[:, :, :bands] = data
        new_data[:, :, bands] = psri

        new_band_names = self.band_names + ['PSRI']
        return new_data, new_band_names

    def REP(self, data, v):
        """
        Compute Red Edge Position (REP) using the Guyot & Baret approximation.
        v: [red_band_index, red_edge_band_index, nir_band_index]
        
        Assumes input data are Planet DN values and converts to reflectance by dividing by 10000.
        """
        band_red = v[0]
        band_re = v[1]
        band_nir = v[2]

        # Convert DN to reflectance (0-1 scale)
        red = data[:, :, band_red].astype(np.float32) / 10000.0
        re = data[:, :, band_re].astype(np.float32) / 10000.0
        nir = data[:, :, band_nir].astype(np.float32) / 10000.0

        # Mask out invalid data
        mask = (red == self.no_data_value) | (re == self.no_data_value) | (nir == self.no_data_value)

        numerator = (re - red) * 35
        denominator = (nir - red)

        # Avoid division by zero
        safe_denominator = np.where(denominator == 0, np.nan, denominator)
        rep = 705 + (numerator / safe_denominator)

        # Apply mask
        rep[mask | np.isnan(rep)] = self.no_data_value

        # Append REP band
        samples, time, bands = data.shape
        new_data = np.zeros((samples, time, bands + 1), dtype=np.float32)
        new_data[:, :, :bands] = data.astype(np.float32) / 10000.0  # scaled original bands as well
        new_data[:, :, bands] = rep

        new_band_names = self.band_names + ['REP']

        return new_data, new_band_names


    def WBI(self, data, v):
        idx = [band for band in v]
        bands_vals = [data[:, :, i] for i in idx]

        # Construct mask for any no_data_value in the used bands
        mask = np.zeros(bands_vals[0].shape, dtype=bool)
        for b in bands_vals:
            mask |= (b == self.no_data_value)

        numerator = bands_vals[0] + bands_vals[1] + bands_vals[2] + bands_vals[5]
        denominator = bands_vals[8] + bands_vals[9] + 1

        wbi = numerator / denominator
        wbi[mask] = self.no_data_value

        samples, time, bands = data.shape
        new_data = np.zeros((samples, time, bands + 1), dtype=np.float32)
        new_data[:, :, :bands] = data
        new_data[:, :, bands] = wbi

        new_band_names = self.band_names + ['WBI']

        return new_data, new_band_names

    def vis(self, data, index_name, bands_for_index, VIs):
        if index_name == 'NDVI':
            data, features_vi = self.NDVI(data, bands_for_index)
        elif index_name == 'CFI3':
            data, NDVI_TS = self.NDVI(data, VIs['NDVI'])
            data, features_vi = self.CFI3(data, bands_for_index, NDVI_TS)
        elif index_name == 'NDYI':
            data, features_vi = self.NDYI(data, bands_for_index)
        elif index_name == 'CRI':
            data, features_vi = self.CRI(data, bands_for_index)
        elif index_name == 'VHVV':
            data, features_vi = self.VHVV(data, bands_for_index)
        elif index_name == 'BHDI1':
            data, features_vi = self.BHDI1(data, bands_for_index)
        elif index_name == 'BHDI2':
            data, features_vi = self.BHDI2(data, bands_for_index)
        elif index_name == 'BHDI3':
            data, features_vi = self.BHDI3(data, bands_for_index)
        elif index_name == 'BHDI4':
            data, features_vi = self.BHDI4(data, bands_for_index)
        elif index_name == 'WBI':
            data, features_vi = self.WBI(data, bands_for_index)
        elif index_name == 'REP':
            data, features_vi = self.REP(data, bands_for_index)
        elif index_name == 'NDRE':
            data, features_vi = self.NDRE(data, bands_for_index)
        elif index_name == 'CIre':
            data, features_vi = self.CIre(data, bands_for_index)
        elif index_name == 'MTCI':
            data, features_vi = self.CIre(data, bands_for_index)
        elif index_name == 'EVI2':
            data, features_vi = self.CIre(data, bands_for_index)
        elif index_name == 'PRI_proxy':
            data, features_vi = self.CIre(data, bands_for_index)
        elif index_name == 'PSRI':
            data, features_vi = self.CIre(data, bands_for_index)
        else:
            raise ValueError(f"Unknown index_name: {index_name}")

        return data, features_vi

    def make_VIs(self, data, VIs, features, feature_names_map):
        for index_name, bands_for_index in VIs.items():
            data, features_vi = self.vis(data, index_name, bands_for_index, VIs)
            features += features_vi
            feature_names_map[index_name] = index_name

        return data, features, feature_names_map


