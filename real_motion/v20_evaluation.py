"""Formal V20 evaluation accumulators shared by Stage-3/4/5 tools."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .metrics.moving_miou_v2 import DYNAMIC_CLASS_IDS

HORIZONS = tuple(0.5 * (i + 1) for i in range(6))
SEMANTIC_CLASSES = tuple(range(17))
MAIN_INDICES = (1, 3, 5)


def _safe_percent(inter, union):
    i = np.asarray(inter, dtype=np.float64)
    u = np.asarray(union, dtype=np.float64)
    out = np.full(i.shape, np.nan, dtype=np.float64)
    np.divide(i, u, out=out, where=u > 0)
    return 100.0 * out


@dataclass
class SemanticMetricAccumulator:
    occ_inter: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.int64))
    occ_union: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.int64))
    sem_inter: np.ndarray = field(
        default_factory=lambda: np.zeros((6, len(SEMANTIC_CLASSES)), dtype=np.int64)
    )
    sem_union: np.ndarray = field(
        default_factory=lambda: np.zeros((6, len(SEMANTIC_CLASSES)), dtype=np.int64)
    )
    mov_inter: np.ndarray = field(
        default_factory=lambda: np.zeros((6, len(DYNAMIC_CLASS_IDS)), dtype=np.int64)
    )
    mov_union: np.ndarray = field(
        default_factory=lambda: np.zeros((6, len(DYNAMIC_CLASS_IDS)), dtype=np.int64)
    )

    def update(self, horizon_index, pred, gt, moving, *, free_label):
        p = np.asarray(pred)
        g = np.asarray(gt)
        m = np.asarray(moving, dtype=bool)
        if p.shape != g.shape or m.shape != g.shape:
            raise ValueError("V20 metric pred/gt/moving shape mismatch")
        hi = int(horizon_index)
        free = int(free_label)
        label_n = max(free + 1, max(int(x) for x in DYNAMIC_CLASS_IDS) + 1, 18)
        fp = p.reshape(-1).astype(np.int16, copy=False)
        fg = g.reshape(-1).astype(np.int16, copy=False)
        pair = fg * label_n + fp
        conf = np.bincount(
            pair, minlength=label_n * label_n
        ).reshape(label_n, label_n)
        fm = m.reshape(-1)
        mconf = (
            np.bincount(
                pair[fm], minlength=label_n * label_n
            ).reshape(label_n, label_n)
            if bool(fm.any())
            else np.zeros((label_n, label_n), dtype=np.int64)
        )
        self.occ_inter[hi] += int(conf[:free, :free].sum())
        self.occ_union[hi] += int(conf.sum() - conf[free, free])
        diag = np.diag(conf)[: len(SEMANTIC_CLASSES)]
        gt_count = conf[: len(SEMANTIC_CLASSES), :].sum(axis=1)
        pr_count = conf[:, : len(SEMANTIC_CLASSES)].sum(axis=0)
        self.sem_inter[hi] += diag
        self.sem_union[hi] += gt_count + pr_count - diag
        for j, cid in enumerate(DYNAMIC_CLASS_IDS):
            cid = int(cid)
            inter = int(mconf[cid, cid])
            union = int(mconf[cid, :].sum() + mconf[:, cid].sum() - inter)
            self.mov_inter[hi, j] += inter
            self.mov_union[hi, j] += union

    def finalize(self):
        occ = _safe_percent(self.occ_inter, self.occ_union)
        sem = _safe_percent(self.sem_inter, self.sem_union)
        mov = _safe_percent(self.mov_inter, self.mov_union)
        miou = np.nanmean(sem, axis=1)
        mmacro = np.nanmean(mov, axis=1)
        mmicro = _safe_percent(
            self.mov_inter.sum(axis=1), self.mov_union.sum(axis=1)
        )
        per = {}
        for hi, h in enumerate(HORIZONS):
            per[str(h)] = {
                "IoU": float(occ[hi]),
                "mIoU": float(miou[hi]),
                "MovingMacro": float(mmacro[hi]),
                "MovingMicro": float(mmicro[hi]),
            }
        return {
            "IoU": float(np.nanmean(occ)),
            "mIoU": float(np.nanmean(miou)),
            "MovingMacro": float(np.nanmean(mmacro)),
            "MovingMicro": float(np.nanmean(mmicro)),
            "main_1_2_3s": {
                "IoU": float(np.nanmean(occ[list(MAIN_INDICES)])),
                "mIoU": float(np.nanmean(miou[list(MAIN_INDICES)])),
                "MovingMacro": float(np.nanmean(mmacro[list(MAIN_INDICES)])),
                "MovingMicro": float(np.nanmean(mmicro[list(MAIN_INDICES)])),
            },
            "per_horizon": per,
            "per_class_all_six_iou": {
                str(cid): float(
                    _safe_percent(
                        np.asarray([self.sem_inter[:, cid].sum()]),
                        np.asarray([self.sem_union[:, cid].sum()]),
                    )[0]
                )
                for cid in SEMANTIC_CLASSES
            },
        }

    def raw(self):
        return {
            "occ_inter": self.occ_inter.tolist(),
            "occ_union": self.occ_union.tolist(),
            "sem_inter": self.sem_inter.tolist(),
            "sem_union": self.sem_union.tolist(),
            "mov_inter": self.mov_inter.tolist(),
            "mov_union": self.mov_union.tolist(),
        }


def metric_delta(a, b):
    keys = ("IoU", "mIoU", "MovingMacro", "MovingMicro")
    return {
        **{k: float(a[k]) - float(b[k]) for k in keys},
        "main_1_2_3s": {
            k: float(a["main_1_2_3s"][k]) - float(b["main_1_2_3s"][k])
            for k in keys
        },
    }


@dataclass
class AdditionAccumulator:
    candidate_voxels: int = 0
    written_voxels: int = 0
    occupied_tp: int = 0
    free_fp: int = 0
    semantic_correct_tp: int = 0

    def update(self, before, proposal, gt, *, free_label):
        b = np.asarray(before)
        p = np.asarray(proposal)
        g = np.asarray(gt)
        if b.shape != p.shape or b.shape != g.shape:
            raise ValueError("addition-stat shape mismatch")
        candidate = p != int(free_label)
        write = candidate & (b == int(free_label))
        occupied = g != int(free_label)
        tp = write & occupied
        self.candidate_voxels += int(candidate.sum())
        self.written_voxels += int(write.sum())
        self.occupied_tp += int((write & occupied).sum())
        self.free_fp += int((write & ~occupied).sum())
        self.semantic_correct_tp += int((tp & (p == g)).sum())

    def finalize(self):
        return {
            "candidate_voxels": int(self.candidate_voxels),
            "written_voxels": int(self.written_voxels),
            "tp": int(self.occupied_tp),
            "fp": int(self.free_fp),
            "precision": float(
                self.occupied_tp / max(self.occupied_tp + self.free_fp, 1)
            ),
            "semantic_correct_tp": int(self.semantic_correct_tp),
            "semantic_accuracy_on_tp": float(
                self.semantic_correct_tp / max(self.occupied_tp, 1)
            ),
        }


@dataclass
class StaticSubsetAccumulator:
    domain_voxels: int = 0
    gt_positive_voxels: int = 0
    added_voxels: int = 0
    added_tp: int = 0
    added_fp: int = 0
    semantic_correct_tp: int = 0

    def update(self, domain, before, after, gt, *, free_label):
        d = np.asarray(domain, dtype=bool)
        b = np.asarray(before)
        a = np.asarray(after)
        g = np.asarray(gt)
        added = d & (b == int(free_label)) & (a != int(free_label))
        occ = g != int(free_label)
        tp = added & occ
        self.domain_voxels += int(d.sum())
        self.gt_positive_voxels += int((d & occ).sum())
        self.added_voxels += int(added.sum())
        self.added_tp += int(tp.sum())
        self.added_fp += int((added & ~occ).sum())
        self.semantic_correct_tp += int((tp & (a == g)).sum())

    def finalize(self):
        return {
            "domain_voxels": int(self.domain_voxels),
            "gt_positive_voxels": int(self.gt_positive_voxels),
            "added_voxels": int(self.added_voxels),
            "added_tp": int(self.added_tp),
            "added_fp": int(self.added_fp),
            "addition_precision": float(
                self.added_tp / max(self.added_tp + self.added_fp, 1)
            ),
            "positive_recall_from_additions": float(
                self.added_tp / max(self.gt_positive_voxels, 1)
            ),
            "semantic_correct_tp": int(self.semantic_correct_tp),
            "semantic_accuracy_on_added_tp": float(
                self.semantic_correct_tp / max(self.added_tp, 1)
            ),
        }


@dataclass
class BirthMatchAccumulator:
    predicted_birth_queries: int = 0
    gt_birth_instances: int = 0
    hungarian_pairs: int = 0
    distance_matched: int = 0
    class_correct_pairs: int = 0
    matched_center_error_sum_m: float = 0.0

    def update(self, row):
        for key in (
            "predicted_birth_queries",
            "gt_birth_instances",
            "hungarian_pairs",
            "distance_matched",
            "class_correct_pairs",
        ):
            setattr(self, key, int(getattr(self, key)) + int(row.get(key, 0)))
        self.matched_center_error_sum_m += float(
            row.get("matched_center_error_sum_m", 0.0)
        )

    def finalize(self):
        return {
            "predicted_birth_queries": int(self.predicted_birth_queries),
            "gt_birth_instances": int(self.gt_birth_instances),
            "hungarian_pairs": int(self.hungarian_pairs),
            "matched": int(self.distance_matched),
            "precision": float(
                self.distance_matched / max(self.predicted_birth_queries, 1)
            ),
            "recall": float(
                self.distance_matched / max(self.gt_birth_instances, 1)
            ),
            "class_correct_pairs": int(self.class_correct_pairs),
            "mean_center_error_m_on_matched": float(
                self.matched_center_error_sum_m / max(self.distance_matched, 1)
            ),
        }


@dataclass
class DormantExistenceAccumulator:
    supervised_tracks: int = 0
    future_negative_tracks: int = 0
    false_active_negative_tracks: int = 0
    horizon_tp: int = 0
    horizon_fp: int = 0
    horizon_fn: int = 0
    horizon_tn: int = 0

    def update(self, pred_active, target_exists, supervised):
        p = np.asarray(pred_active, dtype=bool)
        t = np.asarray(target_exists, dtype=bool)
        s = np.asarray(supervised, dtype=bool)
        if p.shape != t.shape or p.ndim != 2 or s.shape != (p.shape[0],):
            raise ValueError("Dormant existence metric shape mismatch")
        valid = np.broadcast_to(s[:, None], p.shape)
        self.supervised_tracks += int(s.sum())
        neg_track = s & ~t.any(axis=1)
        self.future_negative_tracks += int(neg_track.sum())
        self.false_active_negative_tracks += int(
            (neg_track & p.any(axis=1)).sum()
        )
        self.horizon_tp += int((valid & p & t).sum())
        self.horizon_fp += int((valid & p & ~t).sum())
        self.horizon_fn += int((valid & ~p & t).sum())
        self.horizon_tn += int((valid & ~p & ~t).sum())

    def finalize(self):
        return {
            "supervised_tracks": int(self.supervised_tracks),
            "future_negative_tracks": int(self.future_negative_tracks),
            "false_active_negative_tracks": int(self.false_active_negative_tracks),
            "negative_track_false_activation_rate": float(
                self.false_active_negative_tracks
                / max(self.future_negative_tracks, 1)
            ),
            "track_horizon_tp": int(self.horizon_tp),
            "track_horizon_fp": int(self.horizon_fp),
            "track_horizon_fn": int(self.horizon_fn),
            "track_horizon_tn": int(self.horizon_tn),
            "track_horizon_precision": float(
                self.horizon_tp / max(self.horizon_tp + self.horizon_fp, 1)
            ),
            "track_horizon_recall": float(
                self.horizon_tp / max(self.horizon_tp + self.horizon_fn, 1)
            ),
        }
