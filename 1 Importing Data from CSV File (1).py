# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 19:37:41 2021

@author: Amit Pandey
"""
import pandas as pd

Data1 = pd.read_csv("Osteosarcoma _ with Tumor Size.csv", na_values=["Blank(s)"])
Test1 = Data1.copy()

print(Test1[:5])