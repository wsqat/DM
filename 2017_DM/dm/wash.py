# -*- coding: utf-8 -*-
import csv
import pandas as pd
import numpy as np
import collections
import time
import codecs,os

import datetime
import threading

# 打标签
import station
import workday


import sys
reload(sys)
# sys.setdefaultencoding('gbk')
sys.setdefaultencoding('utf-8')
# sys.setdefaultencoding('utf_8_sig')
import matplotlib.pyplot as plt
plt.rcdefaults()

from datetime import datetime
import random
import json

# 生成cvs文件，key按照什么排序，同时删除临时文件
def sort_by_key(fname, newfname, key):
    # df = pd.read_csv('novels.csv',header=0,usecols=[0,1,6,7,8])
    # df = pd.read_csv('rank_click_month_after.csv',header=0,usecols=[0,1,6,7,8],low_memory=False)
    df = pd.read_csv(fname, header=0, low_memory=False,encoding = "gb2312")
    lc = pd.DataFrame(df)
    lc = lc.dropna(axis=0)
    new = lc.sort_values(by=[''+key+''], ascending=True)
    # print new
    new.to_csv(newfname, sep=',', encoding='gb2312')
    del_cvs_col(newfname, fname, [0])
    os.remove(newfname)

def del_cvs_col(fname, idxs, delimiter=','):
    newfname = "data/temp1" + str(random.randint(1, 10)) + ".csv"
    with open(fname) as csvin, open(newfname, 'w') as csvout:
        reader = csv.reader(csvin, delimiter=delimiter)
        writer = csv.writer(csvout, delimiter=delimiter)
        rows = (tuple(item for idx, item in enumerate(row) if idx not in idxs) for row in reader)
        writer.writerows(rows)
    os.remove(fname)
    os.rename(newfname, fname)

# 删除指定行
def del_cvs_line(fname, line, delimiter=','):
    newfname = "data/gzdata"+str(random.randint(1, 10))+".csv"
    # print newfname
    with open(fname) as csvin, open(newfname, 'w') as csvout:
        reader = csv.reader(csvin, delimiter=delimiter)
        writer = csv.writer(csvout, delimiter=delimiter)
        rows = []
        for row in reader:
            if reader.line_num == line:
                continue
            rows.append(row)
        writer.writerows(rows)
    os.remove(fname)
    os.rename(newfname, fname)

def collect(lst):
    return dict(collections.Counter(lst))

def unique(lst):
    return dict(zip(*np.unique(lst, return_counts=True)))

# 排序，生成前10个目的站点
def topTenDest(fname,delimiter=','):
    # df = pd.read_csv(fname,header=0,usecols=[0,1,2])
    # # df.iloc[:,[0,1]]
    # lc = pd.DataFrame(df)
    # lc = lc.iloc[:,[0,2]]
    # lc = lc.groupby(by=['DESTINATION_CODE']).sum()
    # # 生成的数据类型是Series,如果进一步需要将其转换为dataframe,可以调用Series中的to_frame()方法.
    # # lc = lc.to_frame()
    # # new = lc.sort_values(by=['DESTINATION_CODE'], ascending=False)
    # # new = new.head(20)
    # print lc
    with open(fname) as csvin:
        reader = csv.reader(csvin, delimiter=delimiter)
        for row in reader:
            # 忽略第一行
            # if reader.line_num == 1:
            #     continue
            print row[3]
            if not "号线".encode('gb2312') in row[3]:
                continue

            # print "号线".encode('gb2312')
            # print "号线"
            # splitSam = u"号线"
            dataArr = row[3].split("号线".encode('gb2312'))
            # dataArr = row[3].split(splitSam)

            line = dataArr[0]
            location = dataArr[1]
            # print line
            # print location

            if not line in wordLine:
                wordLine.append(line)
            if not line in counterLine:
                counterLine[line] = 0
            else:
                counterLine[line] += 1

            if not location in wordLocation:
                # location = location.decode('gb2312')
                wordLocation.append(location)
            if not location in counterLocation:
                counterLocation[location] = 0
            else:
                counterLocation[location] += 1


    counter_line_list = sorted(counterLine.items(), key=lambda x: x[1], reverse=True)
    counter_location_list = sorted(counterLocation.items(), key=lambda x: x[1], reverse=True)

    # print(counter_line_list)
    # print(counter_location_list)
    # counter_line_list = json.dumps(counter_line_list, encoding="UTF-8", ensure_ascii=False)
    # print json.dumps(counter_location_list, encoding="UTF-8", ensure_ascii=False)

    line_label = list(map(lambda x: int(x[0]), counter_line_list))
    line_value = list(map(lambda y: y[1], counter_line_list))
    print line_label
    print line_value
    location_label = list(map(lambda x: x[0], counter_location_list[:10]))
    location_value = list(map(lambda y: int(y[1]), counter_location_list[:10]))

    # print location_label
    print json.dumps(location_label, encoding="gbk", ensure_ascii=False)
    print location_value

    #柱状图
    # X = [0, 1, 2, 3, 4, 5]
    # Y = [222, 42, 455, 664, 454, 334]
    fig = plt.figure()
    plt.bar(line_label, line_value, 0.4, color="green")
    plt.xlabel("地铁线号")
    plt.ylabel("人次")
    plt.title("20150401_上海地铁线路人流情况图")
    plt.show()
    plt.savefig("lineChart.jpg")

# 统计词频率
def countGroup(fname):
    df = pd.read_csv(fname,header=0,usecols=[0,2,3,4,5])
    # df = pd.read_csv(fname)
    df = pd.DataFrame(df)
    group1 = df.groupby('id')
    print group1.count()

def mkDir(path):
    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)

def washdata(fname,gzname,newname,delimiter=','):
    dataheader = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    data_original = pd.read_csv(fname, sep=delimiter, header=0, names=dataheader, skip_blank_lines=True)
    data_original['I'] = 0 # 0表示无故障，1表示是否出现故障
    # del_cvs_col(gzname, [0], delimiter)
    # data_original.apply(matchStr)
    # fun = lambda x: washgzdata(x)
    # data_original['I'] = list(map(fun, data_original))
    print gz_data_original
    for indexs in gz_data_original.index:
        # print "indexs "+str(indexs)
        if indexs == 0:
            continue
        # print indexs
        gzstime = gz_data_original.loc[indexs,'A']
        gzetime = gz_data_original.loc[indexs,'B']
        gzsstation = gz_data_original.loc[indexs,'I']
        for oindexs in data_original.index:
            sstation = data_original.loc[oindexs,'B']
            # print str(gzsstation)+"   "+str(sstation)
            if  gzsstation == sstation:
                # print gzsstation.decode("gbk").encode("utf-8")
                print "station ok"
                stime = data_original.loc[oindexs,'C']
                # flag = cmpTime(stime, gzstime, gzetime)
                if cmpTime(stime, gzstime, gzetime):
                    data_original.loc[oindexs,'I'] = 1
                    print str(gzsstation).decode("gbk").encode("utf-8")+","+data_original.loc[oindexs,'C']+" gz true "
    data_original.to_csv(newname, index=False, header=False)

def cmpTime(date,date2,date3):
    # date = "2015-04-01 13:32:00"
    timeArray = time.strptime(date, "%Y-%m-%d %H:%M:%S")
    timeStamp = int(time.mktime(timeArray))
    # date2 = "2015-04-01 13:36:00"
    timeArray2 = time.strptime(date2, "%Y/%m/%d %H:%M:%S")
    timeStamp2 = int(time.mktime(timeArray2))
    # date3 = "2015-04-01 13:36:00"
    timeArray3 = time.strptime(date3, "%Y/%m/%d %H:%M:%S")
    timeStamp3 = int(time.mktime(timeArray3))
    # print timeStamp
    # print timeStamp2
    # print timeStamp3
    if timeStamp>timeStamp2 and timeStamp3>timeStamp:
        print "flag true"
        return 1
    else:
        print "flag false"
        return 0

def washStation(x):
    location = x.strip().decode("gbk").encode("utf-8")
    return station.station(location)

def washLine(fname,newname,delimiter=","):
    dataheader = ['A', 'B', 'C', 'D', 'E', 'F','G']
    data_original = pd.read_csv(fname, sep=delimiter, header=0, names=dataheader, skip_blank_lines=True)
    # sort_by_key(fname, "data/temp/temp_in3" + str(random.randint(1, 1000)) + ".csv", key="B")
    # data_original['H'] = 0  # 0表示无故障，1表示是否出现故障
    data_original['H'] = 1
    fun = lambda x: washStation(x)
    data_original['H'] = list(map(fun, data_original.B))
    data_original.to_csv(newname, index=False ,header=False)
    # del_cvs_col(newname, [0], delimiter)


def calcTime(timeS):
    timeArray = time.strptime(timeS, "%Y-%m-%d %H:%M:%S")
    timeStamp = int(time.mktime(timeArray))
    return timeStamp

def calcTimeH(timeS):
    # timeS = "2015-04-01 05:31:00"
    timeArray = time.strptime(timeS, "%Y-%m-%d %H:%M:%S")
    h = timeArray.tm_hour
    return h

def calcTimeM(timeS):
    timeArray = time.strptime(timeS, "%Y-%m-%d %H:%M:%S")
    s = timeArray.tm_min
    return s

def delMore(pure_name,train_name):
    # A, D, E, F, G, I, K, label
    d = ['line', 'B', 'C', 'count', 'isBusytime', 'isWorkday', 'isIn', 'H', 'isOK']
    # ['line', 'count', 'isBusytime', 'isWorkday', 'isIn', 'isOK', 'isTimes', 'label']
    # A, D, E, F, G, I, hour, min, label
    # data = pd.read_csv(pure_name, sep=",", header=0, names=d, skip_blank_lines=True, encoding="GB2312")
    data = pd.read_csv(pure_name, sep=",", header=0, names=d, skip_blank_lines=True)
    # print data.head(10)
    # 2015-04-01 05:31:00
    # data['D'] = data.C.sp
    fun = lambda x: calcTimeH(x)
    data['hour'] = list(map(fun, data.C))
    funm = lambda x: calcTimeM(x)
    data['min'] = list(map(funm, data.C))
    data['label'] = data['H']
    data.drop('B', axis=1, inplace=True)
    data.drop('C', axis=1, inplace=True)
    data.drop('H', axis=1, inplace=True)
    print data.head(10)
    data.to_csv(train_name, index=False, header=True)

if __name__ == '__main__':
    # dataArr = []
    wordLine = []
    wordLocation = []
    counterLine = {}
    counterLocation = {}
    begin = time.time()

    # dataname = "20150402"
    # dataname = "20150402"
    dataname = "20150407"
    gzfname = "data/gzdata.csv"
    gzdataheader = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    gz_data_original = pd.read_csv(gzfname, sep=",", header=0, names=gzdataheader, skip_blank_lines=True)
    gz_data_original = gz_data_original[gz_data_original['E'].isin(['2015-04-07'])]
    # gz_data_original = gz_data_original[gz_data_original['E'].isin(['2015-04-02'])]
    # fname = "data/SPTCC-20150401_1.csv"
    # fname = "data/SPTCC-" + dataname + ".csv"
    fname = "data/SPTCC-" + dataname + ".csv"
    newname = "data/" + dataname + "_wash.csv"
    station_file = "data/车站.csv"
    final_name = "data/" + dataname + "_step1.csv"
    newname_in = "data/" + dataname + "_step1metro_in.csv"
    newname_out = "data/" + dataname + "_step1metro_out.csv"
    pure_name = "data/"+dataname+"_pure.csv"
    train_name = "data/" + dataname + "_train.bak.csv"

    # step1、时间特征
    # workday.washTime(fname, newname, station_file, final_name, newname_in, newname_out)
    # step2、区域特征
    # washLine(final_name,newname)
    # step3、故障特征
    # washdata(newname,gzfname,pure_name)
    # step 4、删除冗余特征
    # delMore(pure_name, train_name)


    # timeStamp = int(time.mktime(timeArray))
    # print ("当前小时是 %s" % timeStamp.hour)

    folder = "data/"
    filename = folder + "20150401_train.bak_1.csv"
    rst = pd.read_csv(filename, sep=",", header=0, skip_blank_lines=True)
    # for x in range(4,6,1):
        # filename = folder+"2015040"+str(x)+"_pure.csv"
        # filename2 = folder + "2015040" + str(x) + "_train.bak.csv"
        # filename2 = folder + "2015040" + str(x) + "_train.bak.csv"
        # filename3 = folder + "2015040" + str(x) + "_data.csv"
        # delMore(filename, filename2)
        # merge = pd.read_csv(filename2, sep=",", header=0, skip_blank_lines=True)
        # rst = pd.concat([merge, rst])
        # rst = pd.read_csv(filename2, sep=",", skip_blank_lines=True)
        # rst['isWorkday'] = 0
        # print rst.head(10)
        # rst.to_csv(filename2, index=False)
    # print(rst)
    filename2 = folder + "20150404_train.bak_1.csv"
    merge = pd.read_csv(filename2, sep=",", header=0, skip_blank_lines=True)
    rst = pd.concat([merge, rst])
    rst.to_csv("data/201504_data.csv", index=False)
    # rst.to_csv(final_name, encoding='GBK', index=False)


    # getFileNames(folder)
    # calcTime(fname=fname,infile="data/in4.csv",outfile="data/out4.csv",in_out_file="data/in_out4.csv")

    end = time.time()
    print('time is %d seconds ' % (end - begin))
