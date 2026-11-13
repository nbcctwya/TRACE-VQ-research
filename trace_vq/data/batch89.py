"""Full-day Stage1 contrast pairs with exact previous-exchange-day alignment."""
import numpy as np
import pandas as pd

from trace_vq.data.frozen import DailyPositions


class PairedCanonicalDays:
    def __init__(self,source):
        self.source=source;self.days=DailyPositions(source).positions
        index=source.get_index();calendar=source.market_frame.index
        self.day_of_row=np.empty(len(source),dtype=np.int32)
        dates=[index[rows[0]][index.names.index('datetime')] for rows in self.days]
        ordinals=calendar.get_indexer(pd.DatetimeIndex(dates));assert (ordinals>=0).all()
        day_lookup={ordinal:i for i,ordinal in enumerate(ordinals)}
        names=index.get_level_values('instrument').to_numpy();self.previous=[];self.alignments=[]
        for day,rows in enumerate(self.days):
            self.day_of_row[rows]=day;previous=day_lookup.get(ordinals[day]-1)
            if previous is None:
                self.previous.append(day);a=b=np.array([],dtype=np.int64)
            else:
                self.previous.append(previous)
                _,a,b=np.intersect1d(names[rows],names[self.days[previous]],assume_unique=True,return_indices=True)
            self.alignments.append((a.astype(np.int32),b.astype(np.int32)))

    def __len__(self):return len(self.source)
    def get_index(self):return self.source.get_index()
    def __getitem__(self,positions):
        current=self.source[positions]
        if np.ndim(positions)==0:return dict(canonical=current)
        day=int(self.day_of_row[positions[0]])
        if not np.array_equal(positions,self.days[day]):raise ValueError('Complete ordered daily cross-section required')
        a,b=self.alignments[day]
        return dict(canonical=current,previous_canonical=self.source[self.days[self.previous[day]]],align_current=a,align_previous=b)
