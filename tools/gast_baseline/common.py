import hashlib
import io
import json
from contextlib import contextmanager
from pathlib import Path
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
COMMIT='c4687255132e580468e1007168c57b00ce36b76a'
SOURCE_SHA='7760c6f4f92277aa248161cd3e90f87ac2354329babceb6e5ed3b872adde33a0'
FORMAT='GAST_CarryWorld_same_windows_GT_ego_v1'
SHORT=(1,3,5)
LONG=(1,3,5,7,9,11)
GRID=dict(x_min=-40.,y_min=-40.,z_min=-1.,shape_hwd=[200,200,16],voxel_size=[.4,.4,.4])


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(value,indent=2),encoding='utf-8');temp.replace(path)


def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def checked_archive(path):
    with zipfile.ZipFile(path) as z:
        names=z.namelist()
        if len(names)!=len(set(names)):raise ValueError('duplicate asset member')
        m=json.loads(z.read('manifest.json'))
    if (m.get('format')!='carryworld_qualitative_v1' or m.get('preview') is not False
            or m.get('source_sha256')!=SOURCE_SHA or m.get('grid')!=GRID or not m.get('cases')):
        raise ValueError('use completed original CarryWorld qualitative archive / released E / native grid')
    if not any(k.startswith('short') for k in m['cases']) or not any(k.startswith('turn') for k in m['cases']):
        raise ValueError('both short and long comparison cases required')
    for key,c in m['cases'].items():
        short=key.startswith('short');future=c['future_tokens'];hist=c['history_tokens']
        if (not (short or key.startswith('turn')) or len(hist)!=4 or len(future)!=(6 if short else 12)
                or hist[-1]!=c['t0_token'] or len(set(hist+future))!=len(hist+future)):
            raise ValueError('invalid 4+6/12 case identity: '+key)
        if np.asarray(c['future_ego_poses']).shape!=(len(future),4,4):raise ValueError('future poses differ: '+key)
        expected={'t0','gt','c','e'} if short else {'e'}
        if set(c['arrays'])!=expected:raise ValueError('original comparison arrays differ: '+key)
    return m


def select_assets(explicit=None,root=ROOT):
    if explicit:return Path(explicit).expanduser().resolve()
    candidates=set()
    for folder in (root,root/'outputs',root/'outputs/p0_f9_shared_history',root.parent/'_received'):
        if folder.is_dir():
            for pattern in ('*qualitative*.zip','*/*qualitative*.zip','*/*/*qualitative*.zip'):
                candidates.update(p.resolve() for p in folder.glob(pattern) if p.is_file())
    valid=[]
    for path in sorted(candidates):
        try:valid.append((len(checked_archive(path)['cases']),path))
        except (OSError,ValueError,KeyError,zipfile.BadZipFile):continue
    if not valid:raise FileNotFoundError('Set GAST_ASSETS=/absolute/path/to/paper_qualitative_expanded_20261115_231750.zip; no new data export needed')
    maximum=max(n for n,_ in valid);best=[p for n,p in valid if n==maximum]
    if len(best)>1 and len({sha(p) for p in best})>1:
        raise ValueError('Multiple different original archives; set GAST_ASSETS explicitly: '+str(best))
    return best[0]


def array(z,key,name,shape):
    a=np.load(io.BytesIO(z.read(key+'/'+name+'.npy')),allow_pickle=False)
    if a.dtype!=np.uint8 or a.shape!=tuple(shape) or np.any(a>17):raise ValueError('native semantic array differs: '+key+'/'+name)
    return a


def save_npz(path,**values):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+'.tmp')
    with temp.open('wb') as f:np.savez_compressed(f,**values)
    temp.replace(path)


def labels(path,shape=(200,200,16)):
    with np.load(path,allow_pickle=False) as z:a=z['semantics']
    if a.shape!=tuple(shape) or a.dtype.kind not in 'iu' or np.any(a<0) or np.any(a>17):
        raise ValueError('unmodified native 0..17 semantics required: '+str(path))
    return a.astype(np.uint8,copy=False)


def file_inventory(paths):
    return [[str(p),p.stat().st_size,p.stat().st_mtime_ns] for p in sorted(set(map(Path,paths)))]


def verify_inventory(rows):
    for name,size,mtime in rows:
        st=Path(name).stat()
        if (st.st_size,st.st_mtime_ns)!=(size,mtime):raise ValueError('raw selected input changed: '+name)


@contextmanager
def file_lock(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+b') as handle:
        if __import__('os').name=='nt':
            import msvcrt
            handle.write(b'0');handle.flush();handle.seek(0)
            msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
            try:yield
            finally:handle.seek(0);msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)
        else:
            import fcntl
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            try:yield
            finally:fcntl.flock(handle,fcntl.LOCK_UN)
