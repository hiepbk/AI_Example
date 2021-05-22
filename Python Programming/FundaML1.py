#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 17:01:43 2021

@author: hiepbk
"""

#import
import numpy as np
#%%
# x = np.array([[1,2,3],[4,5,6]])
# y = np.ones_like(x) # ma tran toan so mot voi so chieu giong voi x
# np.arange(0, 1, 0.1) ma tran tu 0 den 1 voi cong sai la 0.1
"""
Bài tập 1: Xây dựng mảng các luỹ thừa của 2 nhỏ hơn 1025, 
bao gồm cả 1 = 2**0. Gợi ý: Nếu a là một mảng và b là một 
số thì b**a sẽ trả về một mảng cùng kích thước với a mà phần tử có chỉ 
số i bằng b**a[i], với ** là toán tử luỹ thừa.
"""
x1 = np.arange(0,11,1)
x1 = 2**x1

#%%
"""
Bài tập 2: Xây dựng mảng gồm 10 phần tử, trong đó 9 phần tử đầu bằng 3, phần tử cuối cùng bằng 1.5.
"""
x2 = np.array([3,3,3,3,3,3,3,3,3,1.5])
#%% Chi so nguoc
"""
Để truy cập vào phần tử cuối cùng của mảng này, không cần biết d là bao nhiêu, ta có thể dùng chỉ số -1.
"""
x = np.array([1, 2, 3])
d = x.shape[0]
print(x[-1]) 

#%%
"""
Bài tập: Cho trước một số tự nhiên n. 
Tạo một mảng có n phần tử mà các phần tử có chỉ số chẵn (bắt đầu từ 0) 
là một cấp số cộng bắt đầu từ 2, công sai bằng -0.5; các phần tử có chỉ số lẻ bằng -1.
Ví dụ:
Với n=4, kết quả trả về là mảng [ 2. -1. 1.5 -1. ]. Với n=5, kết quả trả về là mảng [ 2. -1. 1.5 -1. 1. ].
"""
n = 15
cong_sai = 5
ids_chan = np.array([],dtype = int)
ids_le= np.array([],dtype=int)
x3 = np.ones(n,dtype=float)
for i in range(n):
    if i%2==0:
       ids_chan = np.append(ids_chan,i)
    else:
        ids_le =  np.append(ids_le,i)
x3[ids_le] = -1
x3[ids_chan] = np.arange(2, 2+cong_sai*ids_chan.shape[0] , cong_sai)

#%% Phep toan
x4 = np.array([1,2,3])
print( 6/x4) # lay 6 chia cho tung phan tu
print(3**x4) # lay 3 mu tung phan tu
#%%
x = np.array([1, 2, 3])
y = np.array([4, 5, 6])
tich = x*y # nhan tung phan tu voi nhau
sum = np.sum(x)
cos = np.cos(x)
"""
Các hàm toán học trong numpy như: np.abs, np.log, np.exp, np.sin, np.cos, np.tan 
cũng áp dụng lên từng phần tử của mảng. Hàm np.log là logarit tự nhiên, hàm np.exp là hàm e mu x
"""
#%%
"""
Bài tập: Cho một mảng 1 chiều x, tính mảng y và z sao cho 
y[i] = pi/2 - x[i] và z[i] = cos(x[i]) - sin(x[i]). 
Sau đó trả về tổng các phần tử của z
"""
x = np.array([1,2,3,4,5])
y = 3.14/2 - x
z = np.cos(x) - np.sin(x)
sum_z = np.sum(z)
#%%
"""
Bài tập:

Viết hàm số tính tổng trị tuyệt đối các phần tử của một mảng một chiều.

(Gợi ý: np.abs.)
"""
x = np.array([1,2,3,4,5,6,7,8,9,-10])
sum_abs = np.sum(np.abs(x))
#%%
"""
Bài tập: Hãy lập trình hàm softmax.

Gợi ý: Sử dụng hàm np.exp().
"""
def softmax(z):
    a = (np.exp(z))/(np.sum(np.exp(z),axis=0))
    return a
z = np.random.randn(2,3)
a = softmax(z)
sum_1 =np.sum(z,axis=0)
#%% Tich vo huong 
"""
Tích vô hướng (inner product) của hai vectors x và y có cùng số phần tử được định nghĩa như là: 
    np.sum(x*y), tức lấy x nhân với y theo element-wise rồi tính tổng các phần tử:
        Trong numpy su dung np.dot(x,y) de tinh tich vo huong
Bài tập: Tính norm 2 của một vector - vector này được biểu diễn dưới dạng mảng numpy một chiều. Norm 2 của một vector 
x
, được ký hiệu là 
|
|
x
|
|
2
, được định nghĩa là căn bậc hai của tổng bình phương các phần tử của nó.
"""
x = np.array([3,4])
x = np.dot(x,x)
norm_x = np.sqrt(x)
#%%
# a = np.array([5,6,7,3,4,33])
# ids_max =  np.argmax(a)
def softmax(Z):
    """
    Compute softmax values for each sets of scores in V.
    each column of V is a set of score.    
    """
    e_Z = np.exp(Z)
    A = e_Z / e_Z.sum(axis = 0)
    return A
Z = np.array([3, 5, 7, 1, 15, 2000, 1])
A = softmax(Z)
label_max = A.argmax()

#%%
def softmax_stable(Z):
    e_Z = np.exp(Z - np.max(Z, axis = 0, keepdims = True))
    # e_Z = np.exp(Z - 0)
    A = e_Z / e_Z.sum(axis = 0)
    return A
Z = np.array([3, 5, 7, 1, 15, 2000, 1])
A = softmax_stable(Z)
label_max = A.argmax()

    


    

