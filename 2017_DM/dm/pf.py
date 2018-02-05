#encoding: utf-8
# 客流量 3
import pandas as pd
import time as time
from datetime import datetime
import math


# newname_out = "data/step1metro_out.csv"
# newname_in = "data/step1metro_in.csv"
# station_file = "data/车站.csv"
# final_name = "data/step1.csv"

def pf(station_file,newname_in,newname_out,final_name):
    df_in= pd.read_csv(newname_in,encoding='GBK',names=['card_id','time_in','money_in','line_in','station_in','M1_in'])
    df_zuobiao= pd.read_csv(station_file,encoding='GBK')
    print(df_in)

    def t1(g):
        t0 = str(g).strip().split(" ")
        t = t0[1].strip().split(":")
        vector = int(t[0]) * 3600 + int(t[1]) * 60 + int(t[2])
        return vector
    fun=lambda x:(((26940<t1(x)<34260)|(59340<t1(x)<66660))and 1 or 0)



    df_zuobiao0=df_zuobiao.ix[:,['ST_NAME','X','Y']]
    #df_zuobiao0.rename(columns={'ST_NAME':'station_in'}, inplace=True)
    print(df_zuobiao0)
    df_in0=df_in.groupby([df_in['line_in'],df_in['station_in'], df_in['M1_in']])[['card_id']].count()
    df_in0.rename(columns={'card_id':'count'}, inplace=True)
    df_in0.reset_index('M1_in', inplace=True)
    df_in0.reset_index('station_in', inplace=True)
    df_in0.reset_index('line_in', inplace=True)
    df_in0['busytime']=list(map(fun,df_in0['M1_in']))
    df_in0['workday']='1'
    print(df_in0)

    df_out= pd.read_csv(newname_out,encoding='GBK',names=['card_id','time_out','money_out','line_out','station_out','M1_out'])
    df_zuobiao= pd.read_csv(station_file,encoding='GBK')
    print(df_out)
    df_zuobiao0=df_zuobiao.ix[:,['ST_NAME','X','Y']]
    print(df_zuobiao0)
    df_out0=df_out.groupby([df_out['line_out'],df_out['station_out'], df_out['M1_out']])[['card_id']].count()
    df_out0.rename(columns={'card_id':'count'}, inplace=True)
    df_out0.reset_index('M1_out', inplace=True)
    df_out0.reset_index('station_out', inplace=True)
    df_out0.reset_index('line_out', inplace=True)
    df_out0['busytime']=list(map(fun,df_out0['M1_out']))
    df_out0['workday']='1'
    print(df_out0)
    #df_out0.to_csv('/home/suguinan/metroll_out.csv',encoding='GBK', index=False, header=False)
    #df_out1=pd.merge(df_zuobiao0,df_out0,left_on=['ST_NAME'],right_index=True)
    #print(pd.merge(df_zuobiao0,df_out0,left_on=['ST_NAME'],right_index=True))
    frames = [df_in0, df_out0]
    df_out0.columns = ['line','station','M1','count','busytime','workday']
    df_out0['isIn'] = 0
    df_in0.columns = ['line','station','M1','count','busytime','workday']
    df_in0['isIn'] = 1
    result = pd.concat(frames)
    # frames.tocsv("data/step1.csv")
    print(result)
    # pd.concat(frames,axis=1).to_csv(station_file,index=False)
    result.to_csv(final_name,encoding='GBK', index=False ,header=False)
    # file=open(final_name,'w')
    # file.write(str(result))
    # file.close()

    '''df_out1=df_out.groupby([df_out['station_out']])[['card_id']].count()
    df_out1.rename(columns={'card_id':'count'}, inplace=True)
    df_out1.reset_index('station_out', inplace=True)
    print("df_out1 :")
    print(df_out1)
    #df_out2=pd.merge(df_out1,df_zuobiao0,left_on='station_out',right_on='ST_NAME').drop_duplicates()
    #print("df_out2 :")
    #print(df_out2)
    df_out1.to_csv('/media/suguinan/新加卷/DATA/清洗数据/step2metro_outflow.csv',encoding='GBK', index=False ,header=False)
    
    ###########'''


