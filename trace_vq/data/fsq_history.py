"""Label-free FSQ history, causal across stocks, dates and split boundaries."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from trace_vq.data.frozen import atomic_json

FEATURE_DIMS = {"code_delta":12, "code_dwell":3, "code_trans":24, "code_vol":8, "proto_memory":128}
FILES = {"code_delta":"delta", "code_dwell":"dwell", "code_trans":"trans", "code_vol":"vol", "proto_memory":"prototype"}


def index_hash(index):
    return hashlib.sha256(pd.util.hash_pandas_object(index).to_numpy().tobytes()).hexdigest()


def history_features(index, q, read_h, train_mask, calendar, output, *, decay=.99, bins=4096):
    """read_h selects frozen h rows. All prototypes are read before day updates.

    Transition probabilities fit train-only once; other temporal features use
    past market dates. Current q is allowed for deltas, dwell, transition query
    and prototype lookup. N-grams use only previous complete codes.
    """
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    q=np.asarray(q); train_mask=np.asarray(train_mask,dtype=bool)
    assert index.is_unique and q.shape==(len(index),3) and ((q>=0)&(q<8)).all()
    calendar=pd.DatetimeIndex(calendar).sort_values().unique()
    dates=calendar.get_indexer(index.get_level_values("datetime"))
    assert (dates>=0).all()
    stocks,stock_names=pd.factorize(index.get_level_values("instrument"),sort=True)
    grid=np.full((len(calendar),len(stock_names)),-1,dtype=np.int32)
    grid[dates,stocks]=np.arange(len(index),dtype=np.int32)
    order=np.argsort(dates,kind="stable"); boundaries=np.r_[0,np.cumsum(np.bincount(dates,minlength=len(calendar)))]
    day_rows=[order[boundaries[d]:boundaries[d+1]] for d in range(len(calendar))]
    counts=np.ones((3,8,8),dtype=np.int64)  # fixed Laplace smoothing, train only
    for day,rows in enumerate(day_rows):
        if not day or not len(rows): continue
        prev=grid[day-1,stocks[rows]]
        valid=(prev>=0)&train_mask[rows]&train_mask[np.maximum(prev,0)]
        for dimension in range(3):
            np.add.at(counts[dimension],(q[prev[valid],dimension],q[rows[valid],dimension]),1)
    probabilities=counts/counts.sum(-1,keepdims=True)
    dimensions=dict(delta=12,dwell=3,trans=24,vol=8,prototype=read_h(np.array([0])).shape[-1],prototype_mask=1,
                    ngram2=1,ngram3=1)
    arrays={name:np.lib.format.open_memmap(output/(name+".npy"),mode="w+",dtype=np.int32 if name.startswith("ngram") else np.float32,
                                         shape=(len(index),dimension)) for name,dimension in dimensions.items()}
    histogram=np.zeros((len(calendar),27),dtype=np.float32)
    dwell=np.zeros((len(stock_names),3),dtype=np.int32)
    prototypes=np.zeros((512,dimensions["prototype"]),dtype=np.float32)
    seen=np.zeros(512,dtype=bool)
    ids=(q.astype(np.int64)*np.array([1,8,64])).sum(-1)
    for day,rows in enumerate(day_rows):
        if not len(rows): continue
        stock=stocks[rows]; current=q[rows].astype(np.float32)
        prev=grid[max(0,day-1),stock]
        same=(prev>=0)&(day>0)
        prior=q[np.maximum(prev,0)]
        dwell[stock]=np.where(same[:,None]&(prior==q[rows]),dwell[stock]+1,1)
        arrays["dwell"][rows]=np.log1p(np.minimum(dwell[stock],252))/np.log(253)
        pieces=[]; masks=[]
        for lag in (1,5,20):
            past=grid[max(0,day-lag),stock]
            valid=(past>=0)&(day>=lag)
            pieces.append(np.where(valid[:,None],(current-q[np.maximum(past,0)])/7,0))
            masks.append(valid.astype(np.float32))
        arrays["delta"][rows]=np.c_[np.concatenate(pieces,axis=-1),np.stack(masks,axis=-1)]
        arrays["trans"][rows]=np.concatenate([probabilities[j,q[rows,j]] for j in range(3)],axis=-1)
        stds=[]; fractions=[]
        for window in (5,20):
            lags=np.arange(1,window+1)
            past=grid[np.maximum(day-lags,0)[:,None],stock[None,:]]
            valid=(past>=0)&(day>=lags[:,None]); count=valid.sum(0)
            values=q[np.maximum(past,0)].astype(np.float32)/7
            mean=(values*valid[...,None]).sum(0)/np.maximum(count[:,None],1)
            variance=(((values-mean)**2)*valid[...,None]).sum(0)/np.maximum(count[:,None],1)
            stds.append(np.where((count>=2)[:,None],np.sqrt(variance),0))
            fractions.append(count/window)
        arrays["vol"][rows]=np.c_[np.concatenate(stds,axis=-1),np.stack(fractions,axis=-1)]
        for n in (2,3):
            previous=grid[np.maximum(day-np.arange(n,0,-1),0)[:,None],stock[None,:]]
            valid=(previous>=0).all(0)&(day>=n)
            hashed=np.ones(len(rows),dtype=np.int64)
            for source in previous:
                hashed=(hashed*513+ids[np.maximum(source,0)]+1)%(bins-1)
            arrays["ngram"+str(n)][rows,0]=np.where(valid,hashed+1,0)
        arrays["prototype"][rows]=prototypes[ids[rows]]
        arrays["prototype_mask"][rows,0]=seen[ids[rows]]
        h=np.asarray(read_h(rows),dtype=np.float32)
        codes,reverse=np.unique(ids[rows],return_inverse=True)
        sums=np.zeros((len(codes),h.shape[-1]),dtype=np.float64)
        np.add.at(sums,reverse,h)
        mean=(sums/np.bincount(reverse)[:,None]).astype(np.float32)
        prototypes[codes]=np.where(seen[codes,None],decay*prototypes[codes]+(1-decay)*mean,mean)
        seen[codes]=True
        for j in range(3):
            p=np.bincount(q[rows,j],minlength=8)/len(rows)
            histogram[day,j*8:(j+1)*8]=p
            histogram[day,24+j]=-(p*np.log(p+1e-12)).sum()/np.log(8)
    for values in arrays.values(): values.flush()
    np.save(output/"date_codes.npy",dates.astype(np.int32))
    np.save(output/"histogram.npy",histogram)
    np.save(output/"coordinates.npy",q.astype(np.uint8))
    atomic_json(output/"feature_spec.json",dict(index_sha256=index_hash(index),rows=len(index),calendar_days=len(calendar),
        transition_counts=counts.tolist(),transition_probabilities=probabilities.tolist(),transition_fit="train adjacent-market-day observations only",
        delta_lags=[1,5,20],dwell_cap=252,dwell_transform="log1p / log(253)",vol_windows=[5,20],vol_ddof=0,
        vol_semantics="past market days excluding current; missing rows masked, count ratios included",
        ngram_semantics="past 2 or 3 complete codes, excludes current",hash_bins=bins,prototype_decay=decay,
        prototype_semantics="frozen h; previous-date code EMA read first, then update once using whole-day code means; carry across splits"))


class FSQHistoryDataset:
    def __init__(self,source,root,split,variant):
        self.source,self.variant=source,variant
        root=Path(root)
        manifest=json.loads((root/"completion.json").read_text())
        self.start=manifest["offsets"][("train","valid","test").index(split)]
        assert manifest["index_sha256"][split]==index_hash(source.get_index())
        names=[FILES[variant]] if variant in FILES else []
        if variant.startswith("fsq_gram"): names=["ngram"+variant[-1]]
        if variant=="proto_memory": names+=["prototype_mask"]
        self.arrays={name:np.load(root/(name+".npy"),mmap_mode="r") for name in names}
        self.dates=np.load(root/"date_codes.npy",mmap_mode="r") if variant=="code_hist_state" else None
        self.histogram=np.load(root/"histogram.npy",mmap_mode="r") if self.dates is not None else None

    def __len__(self): return len(self.source)
    def get_index(self): return self.source.get_index()

    def __getitem__(self,positions):
        item=self.source[positions]
        rows=self.start+np.asarray(positions)
        if self.variant=="code_hist_state":
            item["state_hist"]=self.histogram[self.dates[rows]].copy()
        elif self.variant.startswith("fsq_gram"):
            item["gram_id"]=self.arrays["ngram"+self.variant[-1]][rows].copy()
        else:
            item["context_feature"]=self.arrays[FILES[self.variant]][rows].copy()
            if self.variant=="proto_memory": item["context_mask"]=self.arrays["prototype_mask"][rows].copy()
        return item
