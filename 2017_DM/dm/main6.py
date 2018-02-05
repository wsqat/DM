# -*- coding: utf-8 -*-
import csv
import pandas as pd
import numpy as np
import collections
import time
import codecs,os

import datetime
import threading

import sys
reload(sys)
# sys.setdefaultencoding('gbk')
sys.setdefaultencoding('utf-8')
# sys.setdefaultencoding('utf_8_sig')
import matplotlib.pyplot as plt
plt.rcdefaults()

import matplotlib.pyplot as plt1
plt1.rcdefaults()

from datetime import datetime
import random

from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['AR PL KaitiM GB']})

import json


# 生成cvs文件, 过滤数据, 清洗数据
# del_cvs_row('20171023.csv', 'rank_click_month_after.csv')
# hotRowNum = 0
def pure_cvs_row(fname, newfname, delimiter=',', key='month'):
    with open(fname) as csvin, open(newfname, 'w') as csvout:
        reader = csv.reader(csvin, delimiter=delimiter)
        # print reader
        # rows = [row for row in reader]
        rows = []
        writer = csv.writer(csvout, delimiter=delimiter)
        for row in reader:
            row_len = len(row)
            if row_len == 9:
                # print row_len
                # print row[0]
                # 加权处理，增加热门作者
                # global hotRowNum
                # hotRowNum += 1
                # if hotRowNum == 1:
                # row[8] = 'month'
                # else:
                # 	# row[8] = str(float(row[6])*0.5 + float(float(row[7])*0.3) + float(row[8])*0.2)
                # 	# row[8] = float(row[6])*0.5 +
                # row[8] = float(row[6])*0.7 + float(row[7])*0.3
                rows.append(row)
        writer.writerows(rows)
    # if key == 'author':
    # 	sort_by_key(newfname,"rank_"+key+".csv",'author')
    # else:
    sort_by_key(newfname, "rank_" + key + ".csv", key)

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

# 生成cvs文件, 删除指定列数
# del_cvs_col(newfname, fname, [0])
def del_cvs_col(fname, newfname, idxs, delimiter=','):
    with open(fname) as csvin, open(newfname, 'w') as csvout:
        reader = csv.reader(csvin, delimiter=delimiter)
        writer = csv.writer(csvout, delimiter=delimiter)
        rows = (tuple(item for idx, item in enumerate(row) if idx not in idxs) for row in reader)
        writer.writerows(rows)

# 删除指定行
def del_cvs_line(fname, line, delimiter=','):
    newfname = "data/data/data1/temp/temp4"+str(random.randint(1, 1000))+".csv"
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

# 按照一卡通卡号，筛选两个站点之间的平均时间
# 1、清洗数据，分为入站数据，出站数据
# 2、按照时间排序
# 3、计算终点站与始发站的时间，写入新的文件
def calcTime(fname,infile,outfile,in_out_file,delimiter=','):

    # 1、清洗数据，分为入站数据，出站数据
    print "---------------------------begin 1、清洗数据 begin---------------------------".encode('gb2312')
    with open(fname) as csvin, open(infile, 'w') as stationIn,open(outfile, 'w') as stationOut:
        reader = csv.reader(csvin, delimiter=delimiter)
        # print reader
        # rows = [row for row in reader]
        rowsIn = [['id','date','time','station','tools','price','promotion']]
        rowsOut = [['id','date','time','station','tools','price','promotion']]
        writerIn = csv.writer(stationIn, delimiter=delimiter)
        writerOut = csv.writer(stationOut, delimiter=delimiter)
        tempPrice = 0
        for row in reader:
            row_len = len(row)
            # print row[3]
            if not "号线".encode('gb2312') in row[3]:
                continue
            # print row
            if float(row[5]) == 0:

                rowsIn.append(row)
            else:
                rowsOut.append(row)

        writerIn.writerows(rowsIn)
        writerOut.writerows(rowsOut)
    csvin.close()
    stationIn.close()
    stationOut.close()
    print "---------------------------end 1、清洗数据 end---------------------------".encode('gb2312')

    # 2、按照时间排序
    print "---------------------------begin 2、时间排序 begin---------------------------".encode('gb2312')
    # sort_by_key(infile, "data/temp_in.csv", key="time")
    # sort_by_key(outfile, "data/temp_out.csv", key="time")
    sort_by_key(infile, "data/data/data1/temp/temp_in3"+str(random.randint(1, 1000))+".csv", key="id")
    sort_by_key(outfile, "data/data/data1/temp/temp_out3"+str(random.randint(1, 1000))+".csv", key="id")
    # sort_by_key(outfile, "data/out3.csv", key="time")
    # outDF = pd.read_csv("data/out3.csv", header=0, delimiter=",", low_memory=False, encoding="gb2312")
    # outdf = pd.DataFrame(outDF)
    # for j, rowOutDf in outdf.iterrows():  # 获取每行的index、row
    #     if j == 0:
    #         continue
    #     # temp = []
    #     # print rowOutDf[2]
    #     # print outdf.loc[j+1,:][2]
    #     # df.loc[[j, j+1], :] = df.loc[[j+1, j], :].values
    #     if rowOutDf[0] != outdf.loc[j+1,:][0]:
    #         continue
    #     if datetime.strptime(rowOutDf[2], '%H:%M:%S') > datetime.strptime(outdf.loc[j+1,:][2], '%H:%M:%S'):
    #         outdf.loc[[j, j+1], :] = outdf.loc[[j+1, j], :].values
    #     #     temp = rowOutDf;
    #     #     rowOutDf = outdf.loc[j+1,:]
    #     #     outdf.loc[j+1, :] = temp
    # outdf.to_csv("data/out3.csv")
    print "---------------------------end 2、时间排序 end---------------------------".encode('gb2312')

    # 3、计算终点站与始发站的时间，写入新的文件
    print "---------------------------begin 3、计算时间 begin---------------------------".encode('gb2312')

    with open(in_out_file, 'w') as stationInOut:
        rowsInOut = [['id','date','stime','etime','time', 'sstation','estation','tools','price','promotion']]
        writerInOut = csv.writer(stationInOut, delimiter=delimiter)

        # pandas
        inDF = pd.read_csv(infile, header=0, delimiter=",", low_memory=False, encoding="gb2312")
        indf = pd.DataFrame(inDF)
        df = pd.read_csv(outfile, header=0, delimiter=",", low_memory=False, encoding="gb2312")
        df = pd.DataFrame(df)
        # for rowIn in readerIn:
        for i, rowIn in indf.iterrows():  # 获取每行的index、row
            # 忽略第一行
            # if readerIn.line_num == 1:
            #     continue
            if i == 0:
                continue
            print "indf.i---" + str(i)
            # row_in_len = len(rowIn)
            # print rowIn
            # id, date, time, station, tools, price, promotion
            id = rowIn[0]
            stime = rowIn[2]
            sstation = rowIn[3]
            row = []
            row.append(id)
            row.append(rowIn[1])
            row.append(stime)

            for index, rowOut in df.iterrows():  # 获取每行的index、row
                if index == 0:
                    continue
                if str(id) != str(rowOut[0]):
                    continue
                # print "id: " + str(id) + " rowOut[0]:" + str(rowOut[0])+" hit"
                # print "index: "+str(index)+" id: " + str(id) + " hit"
                time_s = datetime.strptime(stime, '%H:%M:%S')  # print rowOut[2] 时间字符串转秒数
                time_e = datetime.strptime(rowOut[2], '%H:%M:%S')
                time_cost = (time_e - time_s).seconds
                # print time_cost
                if time_cost > 7200:
                    continue
                m, s = divmod(time_cost, 60)  # print time_cost 秒数转时间字符串
                h, m = divmod(m, 60)
                time_cost = "%d:%02d:%02d" % (h, m, s)
                # row.extend(rowIn[0:2])
                row.append(rowOut[2])
                row.append(time_cost)
                row.append(sstation)
                # row.extend(rowOut[3:6])
                row.append(rowOut[3])
                row.append(rowOut[4])
                row.append(rowOut[5])
                row.append(rowOut[6])
                # print df.shape[0]
                df.drop(index, axis=0, inplace=True)
                break

            # print row
            if len(row) < 6:
                continue
            rowsInOut.append(row)
        writerInOut.writerows(rowsInOut)
    stationInOut.close()
    print "---------------------------end 3、计算时间 end---------------------------".encode('gb2312')



# 扫描指定文件目录下所有csv文件，
# def scan_files(folder):
#     path = os.getcwd()+"/"+folder
#     #获取到当前文件的目录，并检查是否有report文件夹，如果不存在则自动新建report文件
#     files = os.listdir(path)
#     for file in files:
#         if not os.path.isdir(file):# 不是文件夹才打开
#             fname = path+"/"+file
#             print path+"/"+file
#             # topTenDest(fname,delimiter=",")
#             with open(fname,"rb") as csvin:
#                 reader = csv.reader( (line.replace('\0','') for line in csvin), delimiter=",")
#                 for row in reader:
#                     # 忽略第一行
#                     if reader.line_num == 1:
#                         continue
#                     lst.append(int(row[2]))


def mkDir(path):
    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)

# 扫描指定文件目录下所有文件
def getFileNames(folder):
    # 线程池
    threads = []

    path = os.getcwd()+"/"+folder
    #获取到当前文件的目录，并检查是否有report文件夹，如果不存在则自动新建report文件
    files = os.listdir(path)
    # index = 3

    # mkDir(path+"/in")
    # mkDir(path+"/out")
    # mkDir(path+"/in_out")
    # mkDir(path + "/temp")

    # for file in files:
    for index in range(4,100,1):

        fname = path+"SPTCC-20150401_"+ str(index) + ".csv"
        infile = path + "/in/in" + str(index) + ".csv"
        outfile = path + "/out/out" + str(index) + ".csv"
        in_outfile = path + "/in_out/in_out" + str(index) + ".csv"
        print "---------------------------begin " + fname + " begin---------------------------".encode('gb2312')
        calcTime(fname,infile,outfile,in_outfile)
        print "---------------------------end "+fname+" end---------------------------".encode('gb2312')
        # if not os.path.isdir(file):# 不是文件夹才打开
        #     fname = path + "/" + file
        #     if "20150401_" in fname:
        #         # print "exe"
        #         print fname
        #         calcTime(fname,infile,outfile,in_outfile)
                # th = threading.Thread(target=calcTime, args=(fname,infile,outfile,in_outfile))
                # threads.append(th)
                # th.start()
                # time.sleep(1)

    # 等待线程运行完毕
    for th in threads:
        th.join()

if __name__ == '__main__':
    # dataArr = []
    wordLine = []
    wordLocation = []
    counterLine = {}
    counterLocation = {}
    begin = time.time()
    # fname = "data/data/data1/SPTCC-20150401_1.csv"
    folder = "data/data/data1/"
    getFileNames(folder)
    # calcTime(fname=fname,infile="data/in4.csv",outfile="data/out4.csv",in_out_file="data/in_out4.csv")

    end = time.time()
    print('time is %d seconds ' % (end - begin))
    # fname = "data/test.csv"
    # topTenDest(fname=fname)

