"""Opt-in C++ integer kernels. No automatic install/compile in training workers.

SWFM_COLUMN_CPU_BACKEND=numpy (default) keeps the verified NumPy path.
native requires explicit startup prepare_native(); failures do NOT silently
pretend native throughput was measured. ctypes.CDLL releases the GIL.
"""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
from threading import Lock

import numpy as np

ABI = 4
SOURCE = Path(__file__).resolve().parent/'native'/'column_cpu.cpp'
_loaded = None
_load_lock = Lock()


def backend_name():
    name = os.environ.get('SWFM_COLUMN_CPU_BACKEND', 'numpy')
    if name not in ('numpy', 'native'): raise ValueError('SWFM_COLUMN_CPU_BACKEND must be numpy or native')
    return name


def get_native():
    if backend_name() == 'numpy': return None
    return get_prepared_native()


def get_prepared_native():
    """Explicit kernel opt-in without switching every column CPU operation."""
    if _loaded is None: raise RuntimeError('native CPU requested before prepare_native(); run native preflight first')
    return _loaded


def bundle_enabled():
    value = os.environ.get('SWFM_COLUMN_CPU_BUNDLE', '1')
    if value not in ('0', '1'): raise ValueError('SWFM_COLUMN_CPU_BUNDLE must be 0 or 1')
    return backend_name() == 'native' and value == '1'


def _sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _compiler():
    requested = os.environ.get('CXX')
    names = [requested] if requested else (['cl', 'clang++', 'g++'] if os.name == 'nt' else ['c++', 'g++', 'clang++'])
    for name in names:
        candidate = shutil.which(name)
        if not candidate and Path(name).is_file(): candidate = str(Path(name).resolve())
        if candidate: return str(Path(candidate).resolve())
    raise RuntimeError('No C++ compiler found (c++/g++/clang++, or Windows cl). '
        'No package was installed. Use SWFM_COLUMN_CPU_BACKEND=numpy for the existing path.')


def prepare_native(build_dir=None):
    """Compile once on the caller, atomically cache and verify ABI/content.

    No Python/NumPy headers, Ninja, Torch extension or CUDA toolkit required.
    Existing corrupt cache files are never silently overwritten or loaded.
    """
    global _loaded
    with _load_lock:
        if _loaded is not None: return _loaded.info()
        compiler = _compiler(); msvc = Path(compiler).name.lower() in ('cl', 'cl.exe')
        flags = (['/LD', '/O2', '/std:c++17', '/GS-', '/Zl', '/fp:strict'] if msvc else
            ['-shared', '-O3', '-std=c++17', '-fPIC', '-fno-fast-math', '-ffp-contract=off'])
        version = subprocess.run([compiler] if msvc else [compiler, '--version'],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            encoding='mbcs' if os.name == 'nt' else 'utf-8', errors='replace', timeout=15).stdout
        contract = dict(abi=ABI, source_sha256=_sha(SOURCE), compiler=compiler, compiler_version=version,
            flags=flags, machine=platform.machine(), system=platform.system(), pointer_bytes=ctypes.sizeof(ctypes.c_void_p))
        key = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
        root = Path(build_dir or os.environ.get('SWFM_COLUMN_CPU_BUILD_DIR',
            str(SOURCE.parents[2]/'outputs'/'p0_f9_joint_causal_columns'/'native_cpu_cache'))).resolve()
        root.mkdir(parents=True, exist_ok=True)
        directory = root/key; directory.mkdir(exist_ok=True)
        target = directory/('column_cpu.dll' if os.name == 'nt' else 'column_cpu.so')
        manifest_path = directory/'build.json'
        if not target.exists():
            if manifest_path.exists(): raise RuntimeError('incomplete native build cache; use a NEW build directory')
            with tempfile.TemporaryDirectory(prefix='build-', dir=directory) as temp:
                temp = Path(temp); output = temp/target.name
                env = os.environ.copy()
                if msvc:
                    # Pure C ABI integer code, no CRT/DllMain/SDK dependency.
                    env['PATH'] = str(Path(compiler).parent)+os.pathsep+env.get('PATH', '')
                    command = [compiler, *flags, str(SOURCE), '/Fo:'+str(temp/'column_cpu.obj'), '/Fe:'+str(output),
                        '/link', '/NOENTRY', '/NODEFAULTLIB', '/IMPLIB:'+str(temp/'column_cpu.lib')]
                else: command = [compiler, *flags, str(SOURCE), '-o', str(output)]
                result = subprocess.run(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, encoding='mbcs' if os.name == 'nt' else 'utf-8', errors='replace', timeout=120)
                if result.returncode or not output.is_file():
                    raise RuntimeError('native CPU compilation failed; NumPy backend remains available:\n'+result.stdout[-12000:])
                manifest = dict(**contract, fingerprint=key, library_sha256=_sha(output))
                # Publish before loading: Windows cannot rename an in-use DLL.
                # Manifest is published only after the ABI has been checked.
                os.replace(output, target)
                loaded = NativeColumns(target, manifest)
                draft = temp/'build.json'; draft.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
                os.replace(draft, manifest_path)
        if not manifest_path.is_file(): raise RuntimeError('native build manifest missing; use a NEW build directory')
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if any(manifest.get(k) != v for k, v in contract.items()) or manifest.get('library_sha256') != _sha(target):
            raise RuntimeError('native CPU artifact fingerprint mismatch; refusing to load')
        _loaded = loaded if 'loaded' in locals() else NativeColumns(target, manifest)
        return _loaded.info()


def _array(value, dtype, shape=None):
    array = np.asarray(value)
    if array.dtype != np.dtype(dtype): raise TypeError('native array dtype mismatch: '+str(array.dtype)+' != '+str(np.dtype(dtype)))
    if shape is not None and array.shape != tuple(shape): raise ValueError('native array shape mismatch')
    return np.require(array, requirements=['C', 'A'])


def _bits(value, shape=None):
    array = np.asarray(value)
    if array.dtype not in (np.dtype(bool), np.dtype(np.uint8)): raise TypeError('native bits require bool/uint8')
    if shape is not None and array.shape != tuple(shape): raise ValueError('native bits shape mismatch')
    return np.require(array, requirements=['C', 'A']).view(np.uint8)


def _pointer(array): return ctypes.c_void_p(array.ctypes.data)


class NativeColumns:
    """Immutable library; every call owns its inputs/outputs and scratch."""
    def __init__(self, path, manifest):
        self.path, self.manifest = str(Path(path).resolve()), manifest
        self.lib = ctypes.CDLL(self.path)
        self.lib.swfm_column_cpu_abi.argtypes = []; self.lib.swfm_column_cpu_abi.restype = ctypes.c_int
        if self.lib.swfm_column_cpu_abi() != ABI: raise RuntimeError('native column ABI mismatch')
        P, I, J = ctypes.c_void_p, ctypes.c_int64, ctypes.c_int32
        self.rows_fn = self._bind('swfm_rows', [P]*6+[I]*4+[J]*2+[P]*5)
        self.targets_fn = self._bind('swfm_targets', [P]*6+[I]*3+[P])
        self.generation_fn = self._bind('swfm_generation', [P]*2+[I]*4+[P])
        self.static_fn = self._bind('swfm_static', [P]*4+[I]*3+[P], count=True)
        self.static_roi_fn = self._bind('swfm_static_roi', [P,I]+[P]*3+[I]*3+[P])
        self.support_fn = self._bind('swfm_support', [P]+[I]*4+[P]*3, count=True)
        self.gather_fn = self._bind('swfm_gather', [P,I,P,P,I,I,I,J,P,I,P,I,I,P,P])
        self.expand_fn = self._bind('swfm_expand', [P]*4+[I]*3+[J]+[P]*2)
        self.patch_rows_fn = self._bind('swfm_patch_rows', [P]*5+[I]*7+[J]+[P]*2)
        self.changed_fn = self._bind('swfm_changed', [P,I,I,P])
        self.strata_fn = self._bind('swfm_sampling_strata', [P,P,P,I,P,P])
        self.compact_fn = self._bind('swfm_compact_columns', [P]*9+[I]*4+[J]+[P]*8)
        self.support_many_fn = self._bind('swfm_support_many', [P]*2+[I]*5+[P,I]+[P]*3, count=True)
        self.gather_many_fn = self._bind('swfm_gather_many', [P]+[I]*3+[P]*2+[I]*3+[P]*9)
        self.v18_majority_fn = self._bind('swfm_v18_majority', [P]*2+[I]*3+[P]*2)
        self.lock = Lock(); self.calls = {}

    def _bind(self, name, args, count=False):
        fn = getattr(self.lib, name); fn.argtypes = args
        fn.restype = ctypes.c_int64 if count else ctypes.c_int
        return fn

    def _call(self, name, fn, *args):
        code = fn(*args)
        if code < 0: raise ValueError(f'native {name} rejected invalid dimensions/indices ({code})')
        with self.lock: self.calls[name] = self.calls.get(name, 0)+1
        return int(code)

    def info(self):
        with self.lock: calls = dict(self.calls)
        return dict(backend='native', path=self.path, **self.manifest, calls=calls,
            floating_point_geometry='unchanged_numpy_float64', internal_threads=1)

    def v18_majority(self, semantics, unknown):
        shape = self._grid(np.asarray(semantics).shape)
        sem = _array(semantics, np.uint8, shape)
        unknown = _bits(unknown, shape)
        output = np.empty(shape, np.uint8); ambiguous = np.empty(shape, np.uint8)
        self._call('v18_majority', self.v18_majority_fn,
                   _pointer(sem), _pointer(unknown), *shape,
                   _pointer(output), _pointer(ambiguous))
        return output, ambiguous

    @staticmethod
    def _grid(value):
        shape = tuple(int(v) for v in value)
        if len(shape) != 3 or min(shape) < 1 or int(np.prod(shape, dtype=object)) > np.iinfo(np.int64).max:
            raise ValueError('invalid native grid')
        return shape

    def rows(self, xy, classes, allowed, baseline, owners, restored, kind, actor):
        shape = self._grid(np.asarray(baseline).shape); n = len(xy); z = shape[2]
        xy = np.require(xy, dtype=np.int64, requirements=['C', 'A'])
        if xy.shape != (n, 2): raise ValueError('native XY must be N,2')
        classes = _array(classes, np.uint8, (n,)); allowed = _bits(allowed, (n,z))
        baseline = _array(baseline, np.uint8, shape); owners = _array(owners, np.int32, shape)
        restored = _array(restored, np.uint8, shape)
        flat = np.empty((n,z), np.int64); base = np.empty((n,z), np.uint8); fallback = np.empty_like(base)
        legal = np.empty((n,z,3), bool); active = np.empty(n, bool)
        self._call('rows', self.rows_fn, *map(_pointer,(xy,classes,allowed,baseline,owners,restored)),
            n,*shape,kind,actor,*map(_pointer,(flat,base,fallback,legal,active)))
        return flat,base,fallback,legal,active

    def targets(self, plan, gt):
        n,z = plan.base.shape
        arrays = (_array(plan.flat,np.int64,(n,z)), _array(plan.classes,np.uint8,(n,)),
            _array(plan.base,np.uint8,(n,z)), _array(plan.fallback,np.uint8,(n,z)),
            _bits(plan.legal,(n,z,3)), _array(gt,np.uint8))
        labels = np.empty((n,z), np.int64)
        self._call('targets',self.targets_fn,*map(_pointer,arrays),n,z,arrays[-1].size,_pointer(labels))
        return labels

    def generation(self, potential, baseline):
        shape=self._grid(np.asarray(baseline).shape); n=len(potential)
        xy=np.require(potential,dtype=np.int64,requirements=['C','A'])
        if xy.shape != (n,2): raise ValueError('native generation XY shape mismatch')
        baseline=_array(baseline,np.uint8,shape); active=np.empty(n,bool)
        self._call('generation',self.generation_fn,_pointer(xy),_pointer(baseline),n,*shape,_pointer(active))
        return xy[active]

    def static(self, footprint, historical, memory, baseline):
        shape=self._grid(np.asarray(baseline).shape)
        footprint=_bits(footprint,shape[:2]); historical=_bits(historical,shape)
        memory=_array(memory,np.uint8,shape); baseline=_array(baseline,np.uint8,shape)
        xy=np.empty((shape[0]*shape[1],2),np.int64)
        count=self._call('static',self.static_fn,*map(_pointer,(footprint,historical,memory,baseline)),*shape,_pointer(xy))
        return xy[:count]

    def static_roi(self, potential, historical, memory, baseline):
        shape = self._grid(np.asarray(baseline).shape)
        xy = _array(potential, np.int32)
        if xy.ndim != 2 or xy.shape[1] != 2: raise ValueError('static ROI must be N,2')
        historical = _bits(historical, shape)
        memory = _array(memory, np.uint8, shape)
        baseline = _array(baseline, np.uint8, shape)
        active = np.empty(len(xy), np.uint8)
        self._call('static_roi', self.static_roi_fn, _pointer(xy), len(xy),
            *map(_pointer, (historical, memory, baseline)), *shape, _pointer(active))
        return xy[active.astype(bool)]

    def support(self, flat, shape, workspace=None):
        shape=self._grid(shape); flat=_array(flat,np.int64)
        if flat.ndim != 1: raise ValueError('native support flat must be 1D')
        if workspace is None: workspace=(np.empty(shape[:2],np.uint8),np.empty((shape[0]*shape[1],2),np.int64),np.empty(2,np.int64))
        scratch,xy,bounds=workspace
        scratch=_array(scratch,np.uint8,shape[:2]); xy=_array(xy,np.int64,(shape[0]*shape[1],2)); bounds=_array(bounds,np.int64,(2,))
        if not all(a.flags.writeable for a in (scratch, xy, bounds)):
            raise ValueError('native scratch/output workspace must be writable')
        count=self._call('support',self.support_fn,_pointer(flat),len(flat),*shape,*map(_pointer,(scratch,xy,bounds)))
        return xy[:count],int(bounds[0]),int(bounds[1])

    def gather(self, indices, history, observed, owned=None, table=None):
        shape=self._grid(np.asarray(history).shape); indices=_array(indices,np.int64)
        if indices.ndim != 2 or indices.shape[1] != 3: raise ValueError('native gather indices must be N,3')
        history=_array(history,np.uint8,shape); observed=_bits(observed,shape); n=len(indices)
        has_owner=owned is not None
        owned=np.empty(0,np.int64) if owned is None else _array(owned,np.int64)
        if owned.ndim != 1: raise ValueError('native owned must be 1D')
        start,bits=(0,np.empty(0,np.uint8)) if table is None else (int(table[0]),_bits(table[1]))
        if bits.ndim != 1 or start < 0: raise ValueError('native membership table must be 1D/nonnegative')
        labels=np.empty(n,np.uint8); flags=np.empty(n,np.uint8)
        self._call('gather',self.gather_fn,_pointer(indices),n,_pointer(history),_pointer(observed),*shape,
            has_owner,_pointer(owned),len(owned),_pointer(bits),start,len(bits),_pointer(labels),_pointer(flags))
        return labels,flags

    def expand(self, history, flags, inverse, classes, static_actor):
        history=_array(history,np.uint8); flags=_array(flags,np.uint8,history.shape)
        if history.ndim < 2: raise ValueError('native expansion needs full rows')
        inverse=_array(inverse,np.int64); n=len(inverse)
        if inverse.ndim != 1: raise ValueError('native inverse must be 1D')
        classes=_array(classes,np.uint8,(n,)); shape=(n,*history.shape[1:])
        out=np.empty(shape,np.uint8); bits=np.empty_like(out)
        row_size=int(np.prod(history.shape[1:],dtype=object))
        self._call('expand',self.expand_fn,*map(_pointer,(history,flags,inverse,classes)),n,len(history),row_size,
            static_actor,_pointer(out),_pointer(bits))
        return out,bits

    def changed(self, targets):
        targets=_array(targets,np.int64)
        if targets.ndim != 2: raise ValueError('native changed needs N,Z targets')
        n,z=targets.shape; changed=np.empty(n,bool)
        self._call('changed',self.changed_fn,_pointer(targets),n,z,_pointer(changed))
        return changed

    def patch_rows(self, history, flags, starts, rows, classes, static_actor, out, out_flags):
        history = _array(history,np.uint8)
        if history.ndim != 4: raise ValueError('patch map must be F,X,Y,Z')
        frames,xs,ys,zs = history.shape
        flags = _array(flags,np.uint8,history.shape)
        rows = _array(rows,np.int64)
        if rows.ndim != 1: raise ValueError('patch rows must be one dimensional')
        n = len(rows); starts = _array(starts,np.int64,(n,2)); classes = _array(classes,np.uint8,(n,))
        # Outputs must be the ORIGINAL buffers, never an implicit aligned copy.
        for value in (out,out_flags):
            if (not isinstance(value,np.ndarray) or value.dtype != np.uint8 or value.ndim != 5
                    or not value.flags.c_contiguous or not value.flags.aligned or not value.flags.writeable):
                raise ValueError('patch output must be a writable contiguous byte batch')
        batch,ff,p,pp,z = out.shape
        if out_flags.shape != out.shape or ff != frames or p != pp or z != zs:
            raise ValueError('patch output shape mismatch')
        if any(np.shares_memory(value,src) for value in (out,out_flags) for src in (history,flags,starts,rows,classes)) or np.shares_memory(out,out_flags):
            raise ValueError('patch input/output buffers must not alias')
        self._call('patch_rows',self.patch_rows_fn,*map(_pointer,(history,flags,starts,rows,classes)),
            n,batch,frames,xs,ys,zs,p,int(static_actor),_pointer(out),_pointer(out_flags))

    def sampling_strata(self, kinds, actors, positive):
        """One packed row buffer; exact original sorted six TRAIN buckets."""
        kinds = _array(kinds, np.uint8)
        if kinds.ndim != 1: raise ValueError('sampling strata kinds must be 1D')
        n = len(kinds)
        actors = _array(actors, np.int32, (n,)); positive = _bits(positive, (n,))
        rows = np.empty(n, np.int64); offsets = np.empty(7, np.int64)
        self._call('sampling_strata', self.strata_fn, *map(_pointer, (kinds, actors, positive)),
                   n, _pointer(rows), _pointer(offsets))
        return tuple(rows[offsets[b]:offsets[b+1]] for b in range(6))

    def compact(self, xy, kinds, actors, classes, masks, baseline, owners, restored, gt, *, materialize=False, prior_counts=False):
        if materialize and prior_counts: raise ValueError('prior scan must not materialize voxel rows')
        shape=self._grid(np.asarray(baseline).shape); n=len(xy); z=shape[2]
        if z > 64: raise ValueError('compact vertical masks support at most 64 bins')
        arrays=(_array(xy,np.int32,(n,2)), _array(kinds,np.uint8,(n,)), _array(actors,np.int32,(n,)),
            _array(classes,np.uint8,(n,)), _array(masks,np.uint64,(n,)), _array(baseline,np.uint8,shape),
            _array(owners,np.int32,shape), _array(restored,np.uint8,shape), _array(gt,np.uint8,shape))
        active=np.empty(n,bool); positive=np.empty(n,bool)
        dense=(np.empty((n,z),np.int64), np.empty((n,z),np.uint8), np.empty((n,z),np.uint8),
            np.empty((n,z,3),bool), np.empty((n,z),np.int64)) if materialize else ()
        output_ptrs=list(map(_pointer,dense)) if materialize else [None]*5
        counts=np.empty(5,np.int64) if prior_counts else None
        self._call('compact_materialize' if materialize else 'compact_prior' if prior_counts else 'compact_scan', self.compact_fn,
            *map(_pointer,arrays),n,*shape,2 if prior_counts else int(materialize),_pointer(active),_pointer(positive),
            *output_ptrs,_pointer(counts) if counts is not None else None)
        return (active,positive,*dense,*((counts,) if prior_counts else ()))

    def support_many(self, groups, shape):
        shape=self._grid(shape); arrays=[_array(g,np.int64) for g in groups]
        if any(a.ndim != 1 for a in arrays): raise ValueError('support groups must contain 1D flat arrays')
        offsets=np.r_[0,np.cumsum([len(a) for a in arrays],dtype=np.int64)]
        flat=np.concatenate(arrays) if arrays else np.empty(0,np.int64)
        capacity=sum(min(shape[0]*shape[1],5*len(a)) for a in arrays)
        stamp=np.empty(shape[:2],np.uint32); xy=np.empty((capacity,2),np.int64)
        rows=np.empty(len(arrays)+1,np.int64); bounds=np.empty((len(arrays),2),np.int64)
        count=self._call('support_many',self.support_many_fn,_pointer(flat),_pointer(offsets),
            len(arrays),len(flat),*shape,_pointer(stamp),capacity,*map(_pointer,(xy,rows,bounds)))
        return xy[:count],rows,bounds

    def gather_many(self, indices, history, observed, members, tables, valid_frames, row_size):
        history=_array(history,np.uint8)
        if history.ndim != 4: raise ValueError('batch history must be F,X,Y,Z')
        frames=len(history); shape=self._grid(history.shape[1:])
        valid=_bits(valid_frames,(frames,))
        if len(indices) != frames: raise ValueError('batch indices must be F,N,3')
        # Independent float64 transforms already produce one integer array per
        # frame. Retain those arrays and pass pointers, not a second F*N*3 copy.
        frame_indices=[None if a is None else _array(a,np.int64) for a in indices]
        points=next((len(a) for a in frame_indices if a is not None),0)
        for f,a in enumerate(frame_indices):
            if a is None:
                if valid[f]: raise ValueError('missing indices for valid historical frame')
                frame_indices[f]=np.empty((0,3),np.int64)
            elif a.shape != (points,3): raise ValueError('batch indices must be F,N,3')
        ip=np.array([a.ctypes.data for a in frame_indices],np.uintp)
        row_size=int(row_size)
        if row_size < 1 or points%row_size: raise ValueError('batch gather rows must divide point count')
        observed=_bits(observed,history.shape)
        if len(members) != frames or len(tables) != frames: raise ValueError('batch membership frame mismatch')
        # Retain every normalized pointed-to array until CDLL returns.
        owned=[np.empty(0,np.int64) if a is None else _array(a,np.int64) for a in members]
        bits=[np.empty(0,np.uint8) if a is None else _bits(a[1]) for a in tables]
        if any(a.ndim != 1 for a in (*owned,*bits)): raise ValueError('batch membership must be 1D')
        starts=np.array([0 if a is None else int(a[0]) for a in tables],np.int64)
        if np.any(starts < 0): raise ValueError('negative membership table start')
        op=np.array([a.ctypes.data for a in owned],np.uintp); tp=np.array([a.ctypes.data for a in bits],np.uintp)
        on=np.array([len(a) for a in owned],np.int64); tn=np.array([len(a) for a in bits],np.int64)
        has=np.array([a is not None for a in members],np.uint8)
        out=np.empty((points//row_size,frames,row_size),np.uint8); flags=np.empty_like(out)
        self._call('gather_many',self.gather_many_fn,_pointer(ip),frames,points,row_size,
            _pointer(history),_pointer(observed),*shape,
            *map(_pointer,(valid,has,op,on,tp,starts,tn,out,flags)))
        return out,flags
