#coding:'gbk'
#create on 10/10/2018
#@author Jw Huang
#DTW_algorithm/TimeSeries
import os
import numpy as np
from numba import jit
def readfile(file,name):
    mydata=open(os.path.join(file,name),'r')
    headline=mydata.readline()
    headline=headline.strip('\n')
    headline=headline.split(',')
    GridIndex=headline.index('OBJECTID_1')
    HourIndex=headline.index('Hour')
    DayIndex=headline.index('Day')

    #A matrix to store Grids frequency at different time
    Grid_frequency_Matrix=np.zeros((4353,744),dtype='int')
    for line in mydata.readlines():
        line=line.strip('\n')
        line=line.split(',')
        Grid_id=int(line[GridIndex])
        Hour=int(line[HourIndex])
        Day=int(line[DayIndex])
        Grid_frequency_Matrix[Grid_id-1][(Day-1)*24+Hour]=Grid_frequency_Matrix[Grid_id-1][(Day-1)*24+Hour]+1
    mydata.close()

    return Grid_frequency_Matrix

@jit(nopython=True,parallel=True)
def ODMatrix(x,y,D1,r,c):
    # Original Distance Matrix
    for i in range(r):
        for j in range(c):
           # D1[i, j] = euclidean(x[i], y[j])
            D1[i,j]=abs(y[j]-x[i])
    return D1

@jit(nopython=True,parallel=True)
def KP_DTW(D0,D1,r,c):
    #Key point, Dynamic shortest distance
    for i in range(r):
        for j in range(c):
            D1[i,j]+=min(D0[i,j],D0[i,j+1],D0[i+1,j])
    return D1[-1,-1]

def DTW(x,y):
    #DTW algorithm based on euclidean distance/
    #init
    r,c=len(x),len(y)
    D0=np.zeros((r+1,c+1))
    D0[0,1:]=np.inf
    D0[1:,0]=np.inf
    D1=D0[1:,1:]#light copy D0

    #Original Distance Matrix
    D1=ODMatrix(x,y,D1,r,c)

    #Key point, Dynamic shortest distance
    # M=D1.copy()
    D1=KP_DTW(D0,D1,r,c)
    return D1


def DTW_algorithm(Series_Matrix):
    Grid_DTW_matrix=np.zeros((4353,4353),dtype='float')
    # shape=Series_Matrix.shape
    for i in range(0,4352):
        x=Series_Matrix[i]
        for j in range(i+1,4353):
            print('difine DTW ', i,j)
            y=Series_Matrix[j]
            distance=DTW(x,y)
            Grid_DTW_matrix[i][j]=distance
    return Grid_DTW_matrix

def WriteCSV(data,output):
    print('Start to write result')
    txt_file=open(output,'w')
    for line in data:

        txt_file.write(str(line)+'\n')
    txt_file.close()

if __name__ == '__main__':
    file=r'E:\Data\UrbanVitality\WuhanTaxiStaypoints\Original dataset'
    name='sp03_joinWHGrid.csv'
    output=r'E:\Data\UrbanVitality\WuhanTaxiStaypoints\CSV_file_output\Grid_DTW_Matrix2.csv'
    Grid_frequency_Matrix=readfile(file,name)
    Grid_DTW_matrix=DTW_algorithm(Grid_frequency_Matrix)
    Grid_DTW_list=Grid_DTW_matrix.tolist()
    WriteCSV(Grid_DTW_list,output)
    # Grid_frequency_Matrix=Grid_frequency_Matrix.tolist()
    # WriteCSV(Grid_frequency_Matrix,output)