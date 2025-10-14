import pandas as pd
import ast

class Trainbased(object):
    def __init__(self):
        pass

    def get_thresholds(self, df, crop_colname, dist_colname):
        df = pd.DataFrame(df)
        threshold_dict = {}
        for group_name, group_data in df.groupby(crop_colname):
            print(group_name, max(group_data[dist_colname]))
            if isinstance(max(group_data[dist_colname]), list):
                threshold_dict[group_name] = max(group_data[dist_colname])[0]
            else:
                threshold_dict[group_name] = max(group_data[dist_colname])
        return threshold_dict

    def apply_threshold(self, df, crop_colname, v, threshold_dict):
        for k, t in threshold_dict.items():
            try:
                df.loc[((df[crop_colname] == k) & (pd.to_numeric(df[v], errors='coerce') >= float(t))), crop_colname] = 'unknown'
            except:
                import ast
                df.loc[((df[crop_colname] == k) & (pd.to_numeric(df[v], errors='coerce') >= float(ast.literal_eval(t)[0]))), crop_colname] = 'unknown'

        return df