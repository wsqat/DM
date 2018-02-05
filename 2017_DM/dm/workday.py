# encoding: utf-8
# one minuts 1
import pandas as pd
import time as time
from datetime import datetime
import math
from datetime import datetime, timedelta
import vt

def t1(g):
    t = g.strip().split(":")
    vector = int(t[0]) * 3600 + int(t[1]) * 60 + int(t[2])
    return (math.ceil(vector / 60) * 60)

def t2(t):
    m, s = divmod(t, 60)
    h, m = divmod(m, 60)
    # time_str = "2015-04-01" + " " + str(int(h)) + " " + str(int(m)) + " " + str(int(s))
    time_str = "2015-04-07" + " " + str(int(h)) + " " + str(int(m)) + " " + str(int(s))
    time = datetime.strptime(time_str, '%Y-%m-%d %H %M %S')
    print time
    return time.strftime('%Y-%m-%d %H:%M:%S')

# fname = "data/data/data1/SPTCC-20150401_1.csv"
# newname = "data/wash1.csv"
# newname_in = "data/step1metro_in.csv"
# newname_out = "data/step1metro_out.csv"
# station_file = "data/车站.csv"
# final_name = "data/step1.csv"

def washTime(fname,newname,station_file,final_name,newname_in,newname_out):

    df = pd.read_csv(fname, encoding='GBK',
                     names=['card_id', 'date', 'time', 'station', 'vehicle', 'money', 'property'],low_memory=False)
    df1 = df.ix[df.vehicle == u'地铁', ['card_id', 'date', 'time', 'station', 'vehicle', 'money', 'property']]

    df2 = pd.concat([df1, df1['station'].str.split(u'号线', expand=True)], axis=1)
    df2.drop('station', axis=1, inplace=True)
    df2.rename(columns={0: 'line', 1: 'station'}, inplace=True)

    df2.drop('vehicle', axis=1, inplace=True)
    df2.drop('property', axis=1, inplace=True)
    df2.drop('date', axis=1, inplace=True)
    fun = lambda x: t2(t1(x))
    df2['M1'] = list(map(fun, df2.time))

    # df2.drop('time',axis=1, inplace=True)
    # df2.rename(columns={'D':'time'}, inplace=True)
    # print(df2)
    # print "workday"
    # print df2.head(10)
    df2.to_csv(newname, encoding='GBK', index=False, header=False)


    # 调用vt
    vt.vf(newname,newname_in,newname_out,station_file,final_name)