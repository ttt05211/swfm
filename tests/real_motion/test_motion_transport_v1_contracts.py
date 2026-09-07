from dataclasses import fields
import inspect
from real_motion.motion_transport_v1.contracts import CausalInputs
from real_motion.motion_transport_v1.model import MotionTransportV1
def test_causal_contract_has_no_future_object_or_gt_fields():
    names={f.name.lower() for f in fields(CausalInputs)};forbidden=('future_semantics','future_gt','gt_box','annotation_velocity','future_instance','instance_id');assert not any(any(tok in n for tok in forbidden) for n in names);sig=inspect.signature(MotionTransportV1.predict);assert 'targets' not in sig.parameters and 'future_gt' not in sig.parameters
