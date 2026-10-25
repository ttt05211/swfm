"""Legacy metadata may contain six tokens; the actual model input is FOUR."""
import numpy as np
from tools.causal_forward_train.data import ForwardPredictor,forward_window


def four_frame_record(record,raw):
    tokens = tuple(record['history_tokens']);future = tuple(record['future_tokens'])
    labels = np.asarray(raw['history_occ']);observed = np.asarray(raw['history_observed'])
    poses = np.asarray(raw['history_poses'])
    if (len(tokens) < 4 or str(tokens[-1]) != str(record['t0_token']) or len(future) != 6
            or labels.ndim != 4 or labels.shape[0] != 4 or observed.shape != labels.shape
            or observed.dtype != bool or poses.shape != (4,4,4)):
        raise ValueError('history contract: metadata_tokens='+str(len(tokens))+
            ', future_tokens='+str(len(future))+', labels_shape='+str(labels.shape)+
            ', observed_shape='+str(observed.shape)+', poses_shape='+str(poses.shape)+
            '; require last-four history metadata, actual FOUR inputs/poses, SIX futures, final token=t0')
    # No frame/pose/timestamp/label change, no source mutation or extra history.
    return dict(record,history_tokens=tokens[-4:])


class RecoveryPredictor(ForwardPredictor):
    def prepare(self,record,raw,*,times=None):
        return super().prepare(four_frame_record(record,raw),raw,times=times)

    def infer(self,row,poses):
        return forward_window(self.body,self.provider,row,poses=poses,fresh=True)
