from ast import Try
from logging import exception
import pandas as pd
import os
import numpy as np
from multiprocessing import Pool

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

if __name__ == '__main__':
    # test_path = os.path.join(os.getcwd(), '..\\Book-Crossing Dataset\Ratings_testX.csv')
    # test = pd.read_csv(test_path, sep=",", encoding='ISO-8859-1')
    # test.to_csv('Save_Ratings_testX.csv')

    result_csv_path = os.path.join(os.getcwd(), '509557023-org.csv')
    result = pd.read_csv(result_csv_path, sep=",")
    total_count = len(result[result['Success'].notnull()])
    success_count = len(result.loc[result['Success']==True])
    print(f"accuracy: {(success_count/total_count): .0%}")
    result['Predict-Book-Rating'] = result['Predict-Book-Rating'].fillna(0)
    result['Predict-Book-Rating'] =  result['Predict-Book-Rating'].astype('int')
    result['Predict-Book-Rating'].to_csv('submit/result2/509557023.csv', index=False, header=False)



