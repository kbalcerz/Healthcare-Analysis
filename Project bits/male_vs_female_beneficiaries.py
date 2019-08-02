import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from google.cloud import storage
import seaborn as sns
import os
import plotly.offline as py
import plotly.graph_objs as go
py.init_notebook_mode(connected=False)


def getTotalAvg(col):
    x = df[col].apply(lambda x : x if isinstance(x,float) else 0)
    avg_x = int(round(x.mean()))
    x = sum(x)
    return x, avg_x   


df = pd.read_csv('Nursing_2014_processed.csv')
male, avg_male = getTotalAvg('Male Beneficiaries')
female, avg_female = getTotalAvg('Female Beneficiaries')

print("Total Male Beneficiaries : " + str(male))
print("Total Female Beneficiaries : " + str(female))
print("Average number of Male Beneficiaries : " + str(avg_male))
print("Average number of Female Beneficiaries : " + str(avg_female))

N = 1
values = [male, female]
ind = np.arange(N)  # the x locations for the groups
width = 0.9      # the width of the bars

fig, ax = plt.subplots()
bar1 = ax.bar(ind, values[0], width, color='darkorange')

bar2 = ax.bar(ind + width + 0.25, values[1], width)

ax.set_ylabel('Number of Beneficiaries')
ax.set_xlabel('Gender')
ax.set_xticks(ticks = np.arange(0,2, step = ind + width+0.25))
ax.set_xticklabels(('Male', 'Female'))


ax.set_title('Total Number of Male and Female Beneficiaries across different facilities')
ax.legend((bar1[0], bar2[0]), ('Men', 'Women'))
x = plt.show()




