"""Native full-Z history memory, exact dense/tiled execution, sparse readout.

Only causal labels and visibility are encoded. Source membership is injected
after geometric lookup. A session owns ONE window/forward graph, never a
cross-update feature cache. No spatial normalization or height pooling.
"""
from dataclasses import dataclass, asdict
import torch
from torch import nn
from .joint_causal_columns import LinkedColumns
from .sparse_column_readout import safe_memory

PROTOCOL = 'p0_f9_native_full_z_shared_evidence_pilot_v1'


@dataclass(frozen=True)
class SharedEvidenceConfig:
    tile: int = 16
    tile_chunk: int = 32
    samples_per_axis: int = 3
    def validate(self):
        if self.tile < 4 or self.tile_chunk < 1 or self.samples_per_axis != 3:
            raise ValueError('invalid shared evidence contract')


class NativeEncoder(nn.Module):
    """Two cheap spatial convolutions independently on every ordered Z bin."""
    def __init__(self, semantic, channels):
        super().__init__(); self.semantic = semantic
        self.visibility = nn.Linear(1, channels, bias=False)
        self.spatial = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.GELU(), nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1, bias=False))
        # Residual identity initialization; old 64-channel patch CNN is NOT
        # claimed to transfer to this new eight-channel native encoder.
        nn.init.zeros_(self.spatial[-1].weight)
        nn.init.zeros_(self.visibility.weight)

    def forward(self, labels, visibility, *, domain=None):
        # [B,H,W,Z] -> [B,H,W,Z,E]. Outside/unknown never means observed free.
        valid = labels != 18
        x = (self.semantic(labels.long())+self.visibility(visibility[..., None].to(self.semantic.weight.dtype)))
        x = x*valid[..., None]
        b,h,w,z,e = x.shape
        y = x.permute(0,3,4,1,2).reshape(b*z,e,h,w)
        if domain is None: domain=torch.ones((b,h,w),device=y.device,dtype=torch.bool)
        mask=domain[:,None,None].expand(b,z,1,h,w).reshape(b*z,1,h,w)
        residual=y
        # Zero OUTSIDE-grid intermediate values too: two padded convolutions
        # otherwise let halo values leak back across the real map boundary.
        for layer in self.spatial:y=layer(y)*mask
        y=residual+y
        return y.reshape(b,z,e,h,w).permute(0,3,4,1,2)


class SharedHistorySession:
    def __init__(self, model, labels, visibility, *, mode='auto'):
        if mode not in ('auto','dense','tiles'): raise ValueError('invalid encoding mode')
        if labels.ndim != 4 or visibility.shape != labels.shape or len(labels) != model.history_frames:
            raise ValueError('strict historical memory shape mismatch')
        self.model,self.labels,self.visibility,self.mode = model,labels,visibility,mode
        self.tiles = {}; self.dense = None; self.encoding_calls = self.encoded_cells = 0
        self.version = tuple(p._version for p in model.parameters())

    def _validate(self):
        if tuple(p._version for p in self.model.parameters()) != self.version:
            raise RuntimeError('shared learned memory cannot survive an optimizer update')

    def _encode(self, labels, visibility, **kwargs):
        # The sampled-union prefill and the later readout must use the SAME
        # native precision. Otherwise training prefill outside autocast quietly
        # teaches a different encoder than CUDA inference.
        with torch.autocast(device_type=labels.device.type,dtype=torch.bfloat16,
                enabled=labels.device.type=='cuda'):
            return self.model.native(labels,visibility,**kwargs)

    def gather(self, ijk, *, execution=None):
        self._validate()
        # [N,T,K,Z,3], including distinct XY per Z under full ego SE(3).
        n,t,k,z,_ = ijk.shape; _,h,w,nz = self.labels.shape
        if t != len(self.labels) or z != nz: raise ValueError('full Z must be retained')
        valid = ((ijk >= 0)&(ijk < ijk.new_tensor((h,w,nz)))).all(-1)
        safe = ijk.clamp_min(0).minimum(ijk.new_tensor((h-1,w-1,nz-1)))
        frames = torch.arange(t,device=ijk.device).view(1,t,1,1).expand(n,t,k,z)
        valid = valid & (self.labels[frames,safe[...,0],safe[...,1],safe[...,2]] != 18)
        tile = self.model.shared_config.tile; th=(h+tile-1)//tile; tw=(w+tile-1)//tile
        keys = (frames*th+safe[...,0]//tile)*tw+safe[...,1]//tile
        needed = torch.unique(keys[valid])
        choose = execution or self.mode
        # Encoding halo cost, not a fixed assumption that a whole map is cheap.
        if choose == 'auto': choose = 'dense' if len(needed)*(tile+4)**2 >= t*h*w else 'tiles'
        if choose == 'dense' and self.dense is None:
            self.dense = self._encode(self.labels,self.visibility)
            self.encoding_calls += 1; self.encoded_cells += t*h*w*nz
        if self.dense is not None:
            features = self.dense[frames,safe[...,0],safe[...,1],safe[...,2]]
        else:
            # Metadata only: query IDs do not carry labels, poses or features.
            wanted = needed.detach().cpu().tolist()
            missing = [v for v in wanted if v not in self.tiles]
            cfg = self.model.shared_config
            for begin in range(0,len(missing),cfg.tile_chunk):
                group = missing[begin:begin+cfg.tile_chunk]
                ids = torch.tensor(group,device=ijk.device)
                ff = ids//(th*tw); tx = ids//tw%th; ty = ids%tw
                offsets = torch.arange(-2,tile+2,device=ijk.device)
                xx=tx[:,None,None]*tile+offsets[None,:,None]
                yy=ty[:,None,None]*tile+offsets[None,None,:]
                inside=(xx>=0)&(xx<h)&(yy>=0)&(yy<w)
                lab=self.labels[ff[:,None,None],xx.clamp(0,h-1),yy.clamp(0,w-1)]
                vis=self.visibility[ff[:,None,None],xx.clamp(0,h-1),yy.clamp(0,w-1)]
                lab=torch.where(inside[...,None],lab,18);vis=vis&inside[...,None]
                encoded=self._encode(lab,vis,domain=inside)[:,2:tile+2,2:tile+2]
                for j,v in enumerate(group): self.tiles[v]=encoded[j]
                self.encoding_calls+=1;self.encoded_cells+=len(group)*(tile+4)**2*nz
            # No detach: repeated uses accumulate gradient in shared encoded tiles.
            order = wanted
            if order:
                memory=torch.stack([self.tiles[v] for v in order])
                slots=torch.searchsorted(needed,keys)
                slots=slots.clamp_max(len(order)-1)
                features=memory[slots,safe[...,0]%tile,safe[...,1]%tile,safe[...,2]]
            else:
                features=self.model.semantic.weight.new_zeros((*valid.shape,self.model.config.semantic_dim))
        return features*valid[...,None],valid

    def audit(self):
        return dict(encoding_calls=self.encoding_calls,encoded_voxel_cells=self.encoded_cells,
                    unique_tiles=len(self.tiles),dense_encoded=self.dense is not None,
                    cache_scope='one_window_one_forward_no_optimizer_crossing')


class SharedEvidenceColumns(LinkedColumns):
    def __init__(self, config, source_dim, *, history_frames=4, shared_config=SharedEvidenceConfig()):
        if history_frames != 4: raise ValueError('new shared pilot requires FOUR historical observations')
        super().__init__(config,source_dim,history_frames=history_frames)
        shared_config.validate();self.shared_config=shared_config
        # Remove unused old patch CNN; do not optimize parameters that never run.
        del self.spatial
        self.native=NativeEncoder(self.semantic,config.semantic_dim)

    @classmethod
    def from_teacher(cls, teacher, **kwargs):
        result=cls(teacher.config,teacher.source_dim,history_frames=teacher.history_frames,**kwargs)
        source=teacher.state_dict(); destination=result.state_dict()
        copied=[]
        for key in destination:
            old='semantic.weight' if key == 'native.semantic.weight' else key
            if old in source and source[old].shape == destination[key].shape:
                destination[key]=source[old].detach().clone();copied.append(key)
        result.load_state_dict(destination,strict=True)
        result.migration_audit=dict(copied=copied,new_native_encoder=True,
            old_patch_CNN_not_transferred=True,mathematically_equivalent_to_teacher=False)
        return result

    def contract(self):
        return dict(protocol=PROTOCOL,columns=asdict(self.config),shared=asdict(self.shared_config),
                    historical_frames=4,full_z=True,positions='original_offsets_minus3_0_plus3',
                    sampling='full_SE3_per_voxel_nearest_no_height_collapse')

    def read(self, session, ijk, flags, base, fallback, context, kind, classes, source_features):
        features,valid=session.gather(ijk)
        return self.read_features(features,valid,flags,base,fallback,context,kind,classes,source_features)

    def read_features(self,features,valid,flags,base,fallback,context,kind,classes,source_features):
        """Pure bounded reader: native feature lookup stays outside CUDA Graph."""
        if flags.shape != valid.shape: raise ValueError('source membership/visibility lookup mismatch')
        bits=torch.stack(((flags&1)!=0,(flags&2)!=0),-1).to(features.dtype)*valid[...,None]
        x=self.column(torch.cat((features,bits),-1).flatten(-2))*valid.any(-1)[...,None]
        n,t,k,d=x.shape
        axis=torch.tensor([-1.,0.,1.],device=x.device,dtype=x.dtype)
        yy,xx=torch.meshgrid(axis,axis,indexing='ij')
        tt=torch.linspace(-1,0,t,device=x.device,dtype=x.dtype)
        coordinates=torch.stack((tt[:,None].expand(t,k),yy.flatten()[None].expand(t,k),
                                 xx.flatten()[None].expand(t,k)),-1)
        x=x.reshape(n,t*k,d)+self.position(coordinates.reshape(1,t*k,3))
        x,invalid=safe_memory(x,~valid.any(-1).reshape(n,t*k))
        q=(self.query(torch.cat((context.float(),self.semantic(base.long()).flatten(1),
            self.semantic(fallback.long()).flatten(1)),1))+self.kind(kind.long())
            +self.classes(classes.long())+self.source_projection(source_features)).unsqueeze(1)
        for block in self.decoder:q=block(q,x,invalid)
        q=self.norm(q[:,0])
        return self.generation(q),self.refinement(q).reshape(n,self.config.z_bins,3)


from .column_execution import ColumnExecution


class SharedReadExecution(ColumnExecution):
    """Reuse executable reader buffers, NEVER history values or predictions.

    Encoding/SE(3) lookup remain fresh per forward. The inherited engine owns
    capture casts, validates all parameter versions and reports eager fallback.
    Caller MUST clone probability buffers before a subsequent graph replay.
    """
    def __init__(self,model,*,graphs=True,max_graphs=2):
        if not isinstance(model,SharedEvidenceColumns):raise ValueError('shared reader model required')
        self.model=model;self.graphs_enabled=graphs;self.reuse=False;self.max_graphs=max_graphs
        self.graphs={};self.failures=[];self.graph_build_seconds=0.;self.replays=self.eager_calls=0
        self.versions=self._signature()

    def _function(self,batch,legal,*,memory=None,invalid=None):
        g,r=self.model.read_features(**batch)
        p=self.model.calibrated_probabilities(g,r,batch['kind'],legal,validate=False)
        return p,(torch.isfinite(g).all()&torch.isfinite(r).all()&torch.isfinite(p).all()
            &torch.isfinite(batch['source_features']).all())
