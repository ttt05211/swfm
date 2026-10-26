"""Isolated raw four-route evaluator; normalized four-frame forward identity."""
import importlib.util
from tools.fpn_unet import evaluation as original
from tools.causal_forward_train.data import ForwardPredictor,forward_window
from tools.causal_forward_recovery.adapter import four_frame_record

spec = importlib.util.spec_from_file_location('tools.fpn_unet._forward_full_raw_evaluation',original.__file__)
runner = importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


class NormalizedForwardPrepare:
    @staticmethod
    def prepare(adapter,record,raw,*,times=None):
        # Original evaluator owns a SimpleNamespace(provider, history), so call
        # the original no-super builder explicitly on the normalized record.
        return ForwardPredictor.prepare(adapter,four_frame_record(record,raw),raw,times=times)


runner.PROTOCOL = 'full_forward_AdaLN_same64_raw_dense_GT_ego_v1'
runner.DirectPredictor = NormalizedForwardPrepare
runner.forward_window = forward_window
evaluate,NAMES = runner.evaluate,runner.NAMES
