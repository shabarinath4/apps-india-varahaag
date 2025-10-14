import numpy as np

class Preprocess:
    def __init__(
      self      
    ):
       pass

    def composite(
            self,
            ts_sliced, 
            composite_mode
    ):
        if composite_mode == 'mean':
            ts_sliced = np.mean(ts_sliced, axis=1, keepdims=True)
        if composite_mode == 'max':
            ts_sliced = np.max(ts_sliced, axis=1, keepdims=True)
        if composite_mode == 'min':
            ts_sliced = np.min(ts_sliced, axis=1, keepdims=True)

        return ts_sliced

    def slice(
            self, 
            data: np.ndarray, 
            ts_slice_width: int,
            ts_step: int,
            composite_slices: bool,
            composite_mode: str,
            mode: str
    ):
        samples, ts_total, bands = data.shape
        slices = []
        
        for s in range(0, ts_total - ts_slice_width, ts_step):
            ts_sliced = data[:, s:s + ts_slice_width, :]

            if composite_slices:
                ts_sliced = self.composite(
                    ts_sliced, 
                    composite_mode
                    )
            
            slices.append(ts_sliced)

        if mode == 'train':
            slices_stacked = np.concatenate(slices, axis=0)
        if mode == 'inference':
            slices_stacked = np.stack(slices, axis=0)

        return slices_stacked


