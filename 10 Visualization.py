# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import seaborn as sb

# Seaborn Library for Plotting Graps %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Data1 = pd.read_csv("Test_Weather.csv")
Test1 = Data1[["Outlook","Temp","Humidity","Windy","Play"]].copy()

#Regression Line
sb.regplot(x=Test1["Temp"], y=Test1["Play"], fit_reg=True, order=3)

sb.lmplot(x="Temp", y="Humidity", data=Test1, hue="Play", palette="Set1", legend = True)

# Histogram Plot
sb.distplot(Test1["Humidity"], bins=7)

# Bar Plot
sb.countplot(x="Temp", data=Test1)

# Grouped Bar Plot
sb.countplot(x="Temp", data=Test1, hue="Humidity")

# Box Plot (Numerical Attributes)
sb.boxplot(y=Test1["Temp"])
sb.boxplot(x=Test1["Temp"], y=Test1["Humidity"])
sb.boxplot(x=Test1["Temp"], y=Test1["Humidity"], hue="Play", data=Test1)

# Pair Wise Plot
sb.pairplot(Test1, kind="scatter", hue="Play")
