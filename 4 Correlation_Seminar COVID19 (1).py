# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 17:24:11 2020

@author: Amit Pandey
"""

import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

#Pearson Correlational analysis for Anscombe dataset

Data3 = pd.read_csv("Anscombe_Data_Set.csv")
Test3 = Data3[["x1","y1","x2","y2","x3","y3","x4","y4"]].copy()
corr_matrix3=Test3.corr(method="pearson")

f_size=(6,5)
f,(ax_x1,ax_x2,ax_x3,ax_x4)=plt.subplots(4,gridspec_kw={"height_ratios":(0.25,0.25,0.25,0.25)},figsize=f_size)
sb.regplot(x=Test3["x1"],y=Test3["y1"],ax=ax_x1,fit_reg=False)
sb.regplot(x=Test3["x2"],y=Test3["y2"],ax=ax_x2,order=2)
sb.regplot(x=Test3["x3"],y=Test3["y3"],ax=ax_x3,order=1)
sb.regplot(x=Test3["x4"],y=Test3["y4"],ax=ax_x4,order=1)

#-------------------------------------------------------------------------------------------------------------------------------------------

#Spearman Rank Correlational analysis for Anscombe dataset

Data4 = pd.read_csv("Anscombe_Data_Set.csv")
Test4 = Data4[["x1","y1","x2","y2","x3","y3","x4","y4"]].copy()
corr_matrix4=Test4.corr(method="spearman")

f_size=(6,5)
f,(ax_x1,ax_x2,ax_x3,ax_x4)=plt.subplots(4,gridspec_kw={"height_ratios":(0.25,0.25,0.25,0.25)},figsize=f_size)
sb.regplot(x=Test4["x1"],y=Test4["y1"],ax=ax_x1,fit_reg=False)
sb.regplot(x=Test4["x2"],y=Test4["y2"],ax=ax_x2,order=2)
sb.regplot(x=Test4["x3"],y=Test4["y3"],ax=ax_x3,order=1)
sb.regplot(x=Test4["x4"],y=Test4["y4"],ax=ax_x4,order=1)

#-------------------------------------------------------------------------------------------------------------------------------------------

#Kendall Rank Correlational analysis for Anscombe dataset

Data5 = pd.read_csv("Anscombe_Data_Set.csv")
Test5 = Data5[["x1","y1","x2","y2","x3","y3","x4","y4"]].copy()
corr_matrix5=Test5.corr(method="kendall")

f_size=(6,5)
f,(ax_x1,ax_x2,ax_x3,ax_x4)=plt.subplots(4,gridspec_kw={"height_ratios":(0.25,0.25,0.25,0.25)},figsize=f_size)
sb.regplot(x=Test5["x1"],y=Test5["y1"],ax=ax_x1,fit_reg=False)
sb.regplot(x=Test5["x2"],y=Test5["y2"],ax=ax_x2,order=2)
sb.regplot(x=Test5["x3"],y=Test5["y3"],ax=ax_x3,order=1)
sb.regplot(x=Test5["x4"],y=Test5["y4"],ax=ax_x4,order=1)

