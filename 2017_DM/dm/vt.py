#encoding: utf-8

# 虚拟换乘坐 2
import pandas as pd
import time as time
from datetime import datetime
import math
import pf

newname = "data/wash3.csv"
newname_out = "data/step1metro_out.csv"
newname_in = "data/step1metro_in.csv"
station_file = "data/车站.csv"
final_name = "data/step1.csv"

def Complement(x, y):
    import numpy as np
    array1 = np.array(x)
    list1 = array1.tolist()
    array2 = np.array(y)
    list2 = array2.tolist()
    def list_to_tuple(t):
        l = []
        for e in t:
            l.append(tuple(e))
        return l
    def tuple_to_list(t):
        l = []
        for e in t:
            l.append(list(e))
        return l
    a = list_to_tuple(list1)
    b = list_to_tuple(list2)
    set3 = set(b).difference(set(a))
    list3 = list(set3)
    list4 = tuple_to_list(list3)
    from pandas import Series, DataFrame
    df1 = DataFrame(list4, columns=x.columns)
    return df1
def t1(g):
    #t0 = g.strip().split(" ")
    t = g.strip().split(":")
    vector = int(t[0]) * 3600 + int(t[1]) * 60 + int(t[2])
    return vector
    #return (math.ceil(vector / 300) * 300)


def vf(newname,newname_in,newname_out,station_file,final_name):
    df2 = pd.read_csv(newname,encoding='GBK',names=['card_id','time','money','line','station','M1'])
    df_in=df2.ix[df2.money==0.0,['card_id','time','money','line','station','M1']]
    print(df_in)
    df_out=df2.ix[df2.money!=0.0,['card_id','time','money','line','station','M1']]
    print(df_out)
    df_out.columns = ['card_id','time_out','money_out','line_out','station_out','M1_out']
    df_in.columns = ['card_id','time_in','money_in','line_in','station_in','M1_in']

    df_outsh34=df_out.ix[(df_out.station_out==u'上海火车站')&((df_out.line_out==4)|(df_out.line_out==3)),['card_id','time_out','money_out','line_out','station_out','M1_out']]
    df_insh1=df_in.ix[(df_in.station_in==u'上海火车站')&(df_in.line_in==1),['card_id','time_in','money','line_in','station_in','M1_in']]
    result34_1= pd.merge(df_outsh34, df_insh1)
    print(df_insh1)
    print(df_outsh34)
    print(result34_1)
    fun=lambda x,y:t1(x)-t1(y)
    #result['D']='ColumnD'
    #t1(result['time_out'])-t1(result['time_in'])<=60*30
    result34_1['duration']=list(map(fun,result34_1['time_in'],result34_1['time_out']))
    #result[map(lambda x:datetime.datetime(x.year,x.month,x.day,x.hours,x.minutes+30,x.seconds),result['time_in'])>=result['time_out']]
    #result.groupby('card_id')
    print(result34_1)
    result0=result34_1.ix[(result34_1.duration<=60*30)&(result34_1.duration>0),['card_id','time_out','money_out','line_out','station_out','M1_out','time_in''money_in','line_in','station_in','M1_in','duration']]
    print(result0)
    #print(result0.groupby(result0['card_id']).agg({'time_in':['time_in'].min(),'card_id':['card_id']}))
    print(result0.groupby(['M1_in','card_id'])[['duration']].min())
    result1=result0.groupby(['M1_in','card_id'])[['duration']].min()
    print("huhu")
    #result1['card_id']=result1.index
    result1.reset_index('M1_in', inplace=True)
    result1.reset_index('card_id', inplace=True)
    result2=pd.merge(result0,result1,right_on=['card_id','duration'],left_on=['card_id','duration'])
    print(pd.merge(result0,result1,right_on=['card_id','duration'],left_on=['card_id','duration']))

    #print(result0['time_in'].groupby(result0['card_id']).min())
    #result1=result0.ix[result0.time_in==result0['duration'].groupby(result0['card_id']).min(),['card_id','money_out','line_out','station_out','time_out','money_in','line_in','station_in','time_in','duration']]


    #frames =[result2.ix[:,['card_id','money_out','line_out','station_out','time_out']], df_out0]
    #ix[:,['card_id','money_out','line_out','station_out','time_out']
    #In [5]: result = pd.concat(frames)
    #df_out1=pd.concat(frames, axis=0)
    df_out0=Complement(result2.ix[:,['card_id','time_out','money_out','line_out','station_out','M1_out']],df_out).drop_duplicates()
    print(df_out0)
    #df_in1=pd.concat([result1['card_id','money_in','line_in','station_in','time_in'],df_in0], axis=0)
    df_in0=Complement(result2.ix[:,['card_id','time_in','money_in','line_in','station_in','M1_in']],df_in).drop_duplicates()
    print(df_in0)

    ###

    df_outsh1=df_out.ix[(df_out.station_out==u'上海火车站')&(df_out.line_out==1),['card_id','time_out','money_out','line_out','station_out','M1_out']]
    df_insh34=df_in.ix[(df_in.station_in==u'上海火车站')&((df_in.line_in==4)|(df_in.line_in==3)),['card_id','time_in','money','line_in','station_in','M1_in']]
    result1_34= pd.merge(df_outsh1, df_insh34)
    print(df_insh34)
    print(df_outsh1)
    print(result1_34)
    fun=lambda x,y:t1(x)-t1(y)
    #result['D']='ColumnD'
    #t1(result['time_out'])-t1(result['time_in'])<=60*30
    result1_34['duration']=list(map(fun,result1_34['time_in'],result1_34['time_out']))
    #result[map(lambda x:datetime.datetime(x.year,x.month,x.day,x.hours,x.minutes+30,x.seconds),result['time_in'])>=result['time_out']]
    #result.groupby('card_id')
    print(result1_34)
    result3=result1_34.ix[(result1_34.duration<=60*30)&(result1_34.duration>0),['card_id','time_out','money_out','line_out','station_out','M1_out','time_in','money_in','line_in','station_in','M1_in','duration']]
    print(result3)
    #print(result0.groupby(result0['card_id']).agg({'time_in':['time_in'].min(),'card_id':['card_id']}))
    print(result3.groupby(['M1_in','card_id'])[['duration']].min())
    result4=result3.groupby(['M1_in','card_id'])[['duration']].min()
    print("huhu")
    #result4['card_id']=result4.index
    result4.reset_index('M1_in', inplace=True)
    result4.reset_index('card_id', inplace=True)
    result5=pd.merge(result3,result4,right_on=['card_id','duration'],left_on=['card_id','duration'])
    print(pd.merge(result3,result4,right_on=['card_id','duration'],left_on=['card_id','duration']))

    #print(result0['time_in'].groupby(result0['card_id']).min())
    #result1=result0.ix[result0.time_in==result0['duration'].groupby(result0['card_id']).min(),['card_id','money_out','line_out','station_out','time_out','money_in','line_in','station_in','time_in','duration']]

    print(result4)
    #frames =[result2.ix[:,['card_id','money_out','line_out','station_out','time_out']], df_out0]
    #ix[:,['card_id','money_out','line_out','station_out','time_out']
    #In [5]: result = pd.concat(frames)
    #df_out1=pd.concat(frames, axis=0)
    df_out1=Complement(result5.ix[:,['card_id','time_out','money_out','line_out','station_out','M1_out']],df_out0).drop_duplicates()
    print("finaout")
    print(df_out1)
    #df_in1=pd.concat([result1['card_id','money_in','line_in','station_in','time_in'],df_in0], axis=0)
    df_in1=Complement(result5.ix[:,['card_id','time_in','money_in','line_in','station_in','M1_in']],df_in0).drop_duplicates()
    print("finain")
    print(df_in1)

    df_out1.to_csv(newname_out,encoding='GBK', index=False, header=False)
    df_in1.to_csv(newname_in,encoding='GBK', index=False, header=False)

    # 调用pf
    pf.pf(station_file,newname_in,newname_out,final_name)