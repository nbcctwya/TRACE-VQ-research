"""Bounded historical retrieval with explicit trading-calendar label maturity.

Candidates are selected by date/code, never by label magnitude. Read each day
before releasing source labels with maturity strictly earlier than that day.
Held-out memory updates simulate available past outcomes; no weights are fitted
on valid/test labels. The original label-free batch03 cache stays unchanged.
"""
from collections import deque
import json
from pathlib import Path

import numpy as np

from trace_vq.data.frozen import atomic_json
from trace_vq.data.fsq_history import index_hash


def neighbor_codes(code):
    q=np.array([code%8,code//8%8,code//64]);result=[code]
    for axis,step in enumerate((1,8,64)):
        if q[axis]>0:result.append(code-step)
        if q[axis]<7:result.append(code+step)
    return result


def build_candidates(dates,q,labels,output,*,days,candidates=16,bucket_capacity=8,horizon=5):
    """Store [day,512,K] global row IDs; -1 masks absent candidates.

    Conservative as-of rule: a day-d source's Ref(close,-5) outcome can enter
    only when d+5 < query_day. This also excludes self and all same-day rows.
    Day indices MUST come from the complete exchange calendar, not stock rows.
    """
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    dates=np.asarray(dates);q=np.asarray(q);labels=np.asarray(labels)
    assert q.shape==(len(dates),3) and labels.shape==(len(dates),)
    assert dates.min()>=0 and dates.max()<days and ((q>=0)&(q<=7)).all()
    ids=(q.astype(np.int64)*np.array([1,8,64])).sum(-1)
    order=np.argsort(dates,kind='stable');bounds=np.r_[0,np.cumsum(np.bincount(dates,minlength=days))]
    rows=[order[bounds[d]:bounds[d+1]] for d in range(days)]
    pools=np.lib.format.open_memmap(output/'candidates.npy',mode='w+',dtype=np.int32,shape=(days,512,candidates));pools[:]=-1
    buckets=[deque(maxlen=bucket_capacity) for _ in range(512)];neighbors=[neighbor_codes(c) for c in range(512)]
    available=0;lookups=0;max_source_lag=None
    for day in range(days):
        release=day-horizon-1
        if release>=0:
            for row in rows[release]:
                if np.isfinite(labels[row]):buckets[ids[row]].append(int(row));available+=1
        for code in np.unique(ids[rows[day]]):
            candidates_here=[r for neighbor in neighbors[code] for r in buckets[neighbor]]
            # Stable ties are positional IDs in the fixed, source-hashed index.
            chosen=sorted(candidates_here,key=lambda r:(dates[r],r),reverse=True)[:candidates]
            if chosen:
                assert (dates[chosen]+horizon<day).all()
                pools[day,code,:len(chosen)]=chosen;lookups+=1
                lag=int(day-dates[chosen].max());max_source_lag=lag if max_source_lag is None else min(max_source_lag,lag)
    pools.flush();np.save(output/'labels.npy',labels.astype(np.float32))
    spec=dict(days=days,rows=len(dates),candidates=candidates,bucket_capacity=bucket_capacity,horizon=horizon,
        rule='source_day + horizon < query_day',released_finite_labels=available,nonempty_day_code_pools=lookups,
        minimum_observed_lag=max_source_lag,selection='recency of same/axis-neighbor code; label magnitudes not used',
        updates='historical mature labels across train/valid/test; no model fitting on heldout labels')
    atomic_json(output/'memory_spec.json',spec);return spec


class HopfieldDataset:
    def __init__(self,source,root,history_root,split,cache_paths):
        self.source=source;root=Path(root);history_root=Path(history_root)
        manifest=json.loads((history_root/'completion.json').read_text());i=('train','valid','test').index(split)
        self.offsets=manifest['offsets'];self.start=self.offsets[i]
        assert manifest['index_sha256'][split]==index_hash(source.get_index())
        completed=json.loads((root/'completion.json').read_text());assert completed['status']=='complete'
        self.dates=np.load(history_root/'date_codes.npy',mmap_mode='r');q=np.load(history_root/'coordinates.npy',mmap_mode='r')
        self.codes=(q.astype(np.int64)*np.array([1,8,64])).sum(-1)
        self.pools=np.load(root/'candidates.npy',mmap_mode='r');self.labels=np.load(root/'labels.npy',mmap_mode='r')
        self.caches=[np.load(p,mmap_mode='r') for p in cache_paths]
        self.horizon=completed['spec']['horizon']

    def __len__(self):return len(self.source)
    def get_index(self):return self.source.get_index()
    def __getitem__(self,positions):
        item=self.source[positions];rows=self.start+np.asarray(positions)
        chosen=self.pools[self.dates[rows],self.codes[rows]];mask=chosen>=0;safe=np.maximum(chosen,0)
        assert ((self.dates[safe]+self.horizon<self.dates[rows][...,None])|~mask).all()
        keys=np.zeros((*chosen.shape,self.caches[0].shape[-1]),dtype=np.float32)
        for i,cache in enumerate(self.caches):
            take=mask&(safe>=self.offsets[i])&(safe<self.offsets[i+1])
            keys[take]=cache[safe[take]-self.offsets[i],0]
        item.update(memory_keys=keys,memory_values=np.where(mask,self.labels[safe],0).astype(np.float32),memory_mask=mask.astype(np.float32))
        return item
