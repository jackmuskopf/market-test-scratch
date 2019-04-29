# Helper libraries
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import style
style.use('dark_background')

import boto3
client = boto3.Session(profile_name='personal').client("dynamodb")
from jmpy.dynamodb import DynamoDB
dynamodb = DynamoDB(client=client)


def stock_prices(symbol, interpolate=True):
    query = dict(table='Price', filter='symbol = :sym', values={":sym":symbol})
    df = pd.DataFrame(dynamodb.query(**query))
    df.date = df.date.apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))
    df.set_index('date',inplace=True)
    if interpolate:
        return interpolate_df(df)
    else:
        return df

def interpolate_df(df):
    for col in df:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df_reindexed = df.reindex(
        pd.date_range(
            start=df.index.min(),
            end=df.index.max(),
            freq='1D'
        )
    ) 
    df = df_reindexed.interpolate(method='linear') 
    return df

def istockp(s,named=False):
    df = stock_prices(s)
    if named:
        return df.rename(columns = {
            c: "{0}_{1}".format(s,c) for c in df.columns
        })
    else:
        return df

def msframe(*s_arr,**kwargs):
    _filter = kwargs.get('_filter', lambda x: True)
    df =  pd.concat([istockp(s, named=True) for s in s_arr], axis=1)
    return df[[c for c in df.columns if _filter(c)]]


def left_windows(series,size,offset=-1):
    as_strided = np.lib.stride_tricks.as_strided
    data = as_strided(series.values, (len(series) - (size - 1), size), (series.values.strides * 2))
    if offset <= 0:
        nd = len(data)
        data = data[:min([nd-1, nd-1+offset])]
        start_ix = max([0,size-1-offset])
        end_ix = len(data) + start_ix
    else:
        data = data[max([0,offset-size+1]):]
        start_ix = max([size - offset - 1, 0])
        end_ix = -offset
    df = pd.DataFrame(index=series.index[start_ix:end_ix], data=data, columns=list(range(-size+offset+1,offset+1)))
    return df

def right_windows(series,size,offset=1):
    return left_windows(series, size, offset=size-1+offset)