#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 22 10:17:08 2021

@author: hiepbk
"""

#%%
import numpy as np

#%%
"""
Bài tập:

Khai báo một mảng numpy hai chiều A mô tả ma trận:

"""
A = np.array([[1,2,3],[4,5,6],[7,8,9]])

#%%
"""
Với một số tự nhiên n, hãy viết hàm trả về ma trận có dạng:
    tức đường chéo phụ ngay dưới đường chéo chính nhận các giá trị từ 1 đến 
n
. Các thành phần là kiểu số nguyên.
"""
n= 10
diag = np.arange(1,n)
diag_matrix = np.diag(diag,k=-1)

#%%
"""
Cho một ma trận A, viết hàm myfunc tính tổng các phần tử 
trên các cột có chỉ số chẵn (0, 2, 4, ...) của ma trận đó. Ví dụ:
"""
A = np.array([[1,2,3],[4,5,6]])
def myfunct(A):
    sums =[];
    for i in range(A.shape[1]):
        if i%2 ==0:
            sum = np.sum(A[:,i])
            sums.append(sum)
    sum = np.array(sums)
    return sums
sums = myfunct(A)
#%%