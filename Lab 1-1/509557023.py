from cProfile import label
from json.tool import main
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

df = pd.read_csv('glass.data.csv', header=None)
print(df)
all_columns = ["Id number", "RI", "Na", "Mg", "Al",
               "Si", "K", "Ca", "Ba", "Fe", "Type of glass"]

# Q1.
print('Q1:')
# observations counts
print(f'observations counts: {len(df)}')
# attributes counts
print(f'attributes counts: {len(df.columns)}')
# show first 10 observations (remove the first id column and add the title row)
df.columns = all_columns
new_df = df.drop(columns=["Id number"])
print(f'first 10 observations: \n{new_df.head(10)}')
print('--------------------------------------')

# Q2.
print('Q2:')
print(f'columns has missing values: {new_df.columns[new_df.isnull().any()]}')
print('--------------------------------------')

# Q3.
print('Q3:')
# get metal oxide columns
main_columns = all_columns[2:-1]
new_df_mean = new_df[main_columns].mean().sort_values(ascending=False)
print(f'the highest mean: {new_df_mean.head(1).index[0]}')
new_df_std = new_df[main_columns].std().sort_values()
print(f'the smallest standard deviation: {new_df_std.head(1).index[0]}')
print('--------------------------------------')

# Q4.
print('Q4:')
# hist = new_df.plot.hist(by="Type of glass")
# plt.show()
bar = new_df.groupby("Type of glass").size().plot(kind='bar')
plt.title('Counts distribution of glass types', fontsize=16)
print('--------------------------------------')

# Q5
print('Q5:')
# pairplot
sns.pairplot(new_df, hue="Type of glass")

# for col_name in main_columns:
#     print(col_name)
#     new_df.boxplot(by="Type of glass", column=[col_name])
#     plt.show()
#     new_df.plot(by="Type of glass", column=[col_name])
#     plt.show()

# boxplot
new_df.boxplot(by="Type of glass")
plt.show()
# correlation plot 
# corr = new_df[main_columns].corr()
# plt.matshow(corr)
# plt.xticks(range(len(corr.columns)), corr.columns)
# plt.yticks(range(len(corr.columns)), corr.columns)
# # print(corr)
# plt.colorbar()
# plt.title('Correlation Matrix', fontsize=16)
# plt.show()
