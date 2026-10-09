"""Explicit, separately fingerprinted native execution; no install/worker build."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
from threading import Lock

import numpy as np
from .native_column_cpu import _compiler, _array, _pointer, _sha

SOURCE = Path(__file__).parent/'native'/'waymo_execution.cpp'
_loaded = None
_lock = Lock()


def prepare_waymo_native(build_dir=None):
    global _loaded
    with _lock:
        if _loaded is not None: return _loaded
        compiler = _compiler(); msvc = Path(compiler).name.lower() in ('cl','cl.exe')
        flags = (['/LD','/O2','/std:c++17','/GS-','/Zl','/fp:strict'] if msvc else
                 ['-shared','-O3','-std=c++17','-fPIC','-fno-fast-math','-ffp-contract=off'])
        version = subprocess.run([compiler] if msvc else [compiler,'--version'],
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,errors='replace',timeout=15).stdout
        contract = dict(abi=1,source_sha256=_sha(SOURCE),compiler=compiler,compiler_version=version,
            flags=flags,machine=platform.machine(),system=platform.system(),pointer_bytes=ctypes.sizeof(ctypes.c_void_p))
        key = hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
        root = Path(build_dir or SOURCE.parents[2]/'outputs'/'waymo_native_cache').resolve()
        directory = root/key; directory.mkdir(parents=True,exist_ok=True)
        target = directory/('waymo_execution.dll' if msvc else 'waymo_execution.so')
        manifest_path = directory/'build.json'
        if not target.exists():
            if manifest_path.exists(): raise RuntimeError('incomplete Waymo kernel artifact; use new build directory')
            # Short same-volume scratch avoids MSVC MAX_PATH; no SDK/CRT needed.
            scratch = SOURCE.parents[2]/'outputs'; scratch.mkdir(exist_ok=True)
            with tempfile.TemporaryDirectory(prefix='waymo-build-',dir=scratch) as temp:
                temp = Path(temp); output = temp/target.name
                command = ([compiler,*flags,str(SOURCE),'/Fo:waymo_execution.obj','/Fe:waymo_execution.dll',
                    '/link','/NOENTRY','/NODEFAULTLIB','/IMPLIB:waymo_execution.lib'] if msvc else
                    [compiler,*flags,str(SOURCE),'-o',str(output)])
                env = os.environ.copy(); env['PATH']=str(Path(compiler).parent)+os.pathsep+env.get('PATH','')
                result = subprocess.run(command,cwd=temp,env=env,stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,text=True,errors='replace',timeout=120)
                if result.returncode or not output.is_file():
                    raise RuntimeError('Waymo native compilation failed; no fallback speed claim:\n'+result.stdout[-12000:])
                # Publication is atomic on this workspace volume, original caches untouched.
                draft = directory/'library.partial'
                with output.open('rb') as src, draft.open('xb') as dst:
                    for block in iter(lambda:src.read(2**20),b''): dst.write(block)
                os.replace(draft,target)
                manifest = dict(**contract,fingerprint=key,library_sha256=_sha(target))
                NativeWaymo(target,manifest)
                draft_manifest = directory/'build.partial.json'
                draft_manifest.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
                os.replace(draft_manifest,manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if any(manifest.get(k)!=v for k,v in contract.items()) or manifest.get('library_sha256')!=_sha(target):
            raise RuntimeError('Waymo native fingerprint mismatch')
        _loaded = NativeWaymo(target,manifest)
        return _loaded


class NativeWaymo:
    def __init__(self,path,manifest):
        self.manifest = manifest; self.path = str(path); self.lib = ctypes.CDLL(self.path)
        self.lib.swfm_waymo_execution_abi.restype = ctypes.c_int
        if self.lib.swfm_waymo_execution_abi()!=1: raise RuntimeError('Waymo native ABI mismatch')
        ptr = ctypes.c_void_p; size = ctypes.c_longlong
        self.warp_fn = self.lib.swfm_warp_winners
        self.warp_fn.argtypes = [ptr,ptr,ptr,size,size,ctypes.c_int,ptr,ptr]; self.warp_fn.restype = ctypes.c_int
        self.fit_fn = self.lib.swfm_surface_fit
        self.fit_fn.argtypes = [ptr,ptr,ptr,ptr,ptr,ptr,size,ptr]; self.fit_fn.restype = ctypes.c_int

    def warp(self,flat,distance,labels,shape,free=17):
        n = len(flat); volume = int(np.prod(shape))
        flat=_array(flat,np.int64,(n,)); distance=_array(distance,np.float64,(n,)); labels=_array(labels,np.uint8,(n,))
        if volume<1 or not 0<=free<=255 or (n and (flat.min()<0 or flat.max()>=volume)):
            raise ValueError('invalid native warp grid/indices')
        if not np.isfinite(distance).all() or np.any(distance<0): raise ValueError('invalid native distances')
        out=np.empty(volume,np.uint8); best=np.empty(volume,np.float64)
        if self.warp_fn(*map(_pointer,(flat,distance,labels)),n,volume,free,_pointer(out),_pointer(best)):
            raise RuntimeError('Waymo native warp failed')
        return out.reshape(shape)

    def fit(self,delta,weight,distance,recent,opposite,opposite_seen):
        n=len(delta)
        arrays=(_array(delta,np.float64,(n,16,3)),_array(weight,np.float64,(n,16)),
            _array(distance,np.float64,(n,16)),_array(recent,np.float64,(n,16)),
            _array(opposite,np.float64,(n,)),_array(opposite_seen,np.uint8,(n,)))
        out=np.empty((n,12),np.float64)
        if self.fit_fn(*map(_pointer,arrays),n,_pointer(out)): raise RuntimeError('Waymo native fit failed')
        out[:,2:4]=np.sqrt(out[:,2:4])
        return np.clip(out,-4.,4.).astype(np.float32)
