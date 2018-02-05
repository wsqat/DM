# -*- coding: utf-8 -*-
import os
import time
import csv
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import pandas as pd

# -*- coding: cp936 -*-
def mkSubFile(lines, head, srcName, sub):
    [des_filename, extname] = os.path.splitext(srcName)
    filename = des_filename + '_' + str(sub) + extname
    print('make file: %s' % filename)
    fout = open(filename, 'w')
    try:
        fout.writelines([head])
        fout.writelines(lines)
        return sub + 1
    finally:
        fout.close()


def splitByLineCount(filename, count):
    fin = open(filename, 'r')
    try:
        head = fin.readline()
        buf = []
        sub = 1
        count = 0
        for line in fin:
            buf.append(line)
            if len(buf) == count:
                sub = mkSubFile(buf, head, filename, sub)
                buf = []
        if len(buf) != 0:
            sub = mkSubFile(buf, head, filename, sub)
    finally:
        fin.close()

# 分割
def splitByLineCountAvg(filename, count):
    fin = open(filename, 'r')
    try:
        head = fin.readline()
        buf1 = []
        buf2 = []
        buf3 = []
        buf4 = []
        buf5 = []
        sub = 1
        total_buf = []
        for line in fin:
            label = int(line.split(',')[8])
            if label == 1:
                buf1.append(line)
            if label == 2:
                buf2.append(line)
            if label == 3:
                buf3.append(line)
            if label == 4:
                buf4.append(line)
            if label == 5:
                buf5.append(line)
            if len(buf1) == count:
                print "buf1 ok"
                total_buf += buf1
                # sub = mkSubFile(buf1, head, filename, sub)
            if len(buf2) == count:
                print "buf2 ok"
                total_buf += buf2
                # sub = mkSubFile(buf2, head, filename, sub)
            if len(buf3) == count:
                print "buf3 ok"
                total_buf += buf3
                # sub = mkSubFile(buf3, head, filename, sub)
            if len(buf4) == count:
                print "buf4 ok"
                total_buf += buf4
                # sub = mkSubFile(buf4, head, filename, sub)
            if len(buf5) == count:
                print "buf5 ok"
                total_buf += buf5
                # sub = mkSubFile(buf5, head, filename, sub)
            if len(total_buf) == 5 * count:
                sub = mkSubFile(total_buf, head, filename, sub)
                break
    finally:
        fin.close()

# 先过滤数据
def washFile(fname,newfile,delimiter=','):
    with open(fname) as csvin, open(newfile, 'w') as csvout:
        reader = csv.reader(csvin, delimiter=delimiter)
        writer = csv.writer(csvout, delimiter=delimiter)
        rows = []
        for row in reader:
            print "reader: "+str(reader.line_num)
            if not "号线".encode('gb2312') in row[3]:
                continue
            rows.append(row)
        writer.writerows(rows)
        print "writer: " + str(rows.__len__())

# 生成cvs文件，key按照什么排序，同时删除临时文件
def sort_by_key(fname, newfname):
    # df = pd.read_csv('novels.csv',header=0,usecols=[0,1,6,7,8])
    # df = pd.read_csv('rank_click_month_after.csv',header=0,usecols=[0,1,6,7,8],low_memory=False)
    # df = pd.read_csv(fname, header=0, low_memory=False,encoding = "gb2312")
    dataheader = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    # gz_data_original = pd.read_csv(gzname, sep=delimiter, header=0, names=gzdataheader, skip_blank_lines=True)
    df = pd.read_csv(fname, header=0, names=dataheader, skip_blank_lines=True,encoding = "gb2312")
    lc = pd.DataFrame(df)
    lc = lc.dropna(axis=0)
    # new = lc.sort_values(by=[''+key+''], ascending=True)
    new = lc.sort_values(by=['C'], ascending=True)
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

if __name__ == '__main__':
    begin = time.time()
    # fname = "data/SPTCC-20150401.csv"
    # reader: 14836174
    # writer: 9024322
    # nname = "data/data/SPTCC-20150401.csv"
    # newfname = "data/data/data/SPTCC-20150401.csv"
    # sort_by_key(nname,newfname)
    fname = "data/20150401_train.bak.csv"
    splitByLineCountAvg(fname, 10000)
    # import pandas as pd
    # df = pd.read_csv(fname,delimiter=",")
    # group1 = df.groupby('label')
    # print group1.count()

    # washFile(fname, nname, delimiter=',')
    # splitByLineCount(nname, 10000)
    end = time.time()
    print('time is %d seconds ' % (end - begin))