from data import Data
from model import Model
from time import time
import pandas as pd
import os
from stack import Stacker

if __name__ == '__main__':
    start = time()
    param_cat = {'cat': {'iterations': 675,
                         'learning_rate': 0.02,
                         'loss_function': 'MAE',
                         'eval_metric': 'MAE',
                         'l2_leaf_reg': 115,
                         'depth': 6,
                         'random_seed': 20}}
    filename = 'catboost'
    for year in [2016, 2017]:
        df = Data(year)
        df.assign_test_year(year)
        model = Model(df, 'cat', param_cat)
        model.create_fit()
        for month in [10, 11, 12]:
            # if month == 10:
            #    xgb.cross_validation_all('mae', 3)
            print('\nStart predicting for month {}'.format(month))
            model.assign_df_month(month)
            pred = model.predict_all()
            df.write_submission(pred, year, month)
            # stack.stack_model_cv('mae', 3)
            # pred = stack.predict(0)
            # df.submit(filename, year)
        if year == 2016:
            f2016 = df.submission
        else:
            f2017 = df.submission
    result = pd.merge(f2016, f2017, on='ParcelId')
    print(result.shape)
    print(result.head())
    if not os.path.exists('output'):
        os.makedirs('output')
    result.to_csv('output/' + filename + '_meta.csv', index=False)
    result.to_csv('output/' + filename + '.csv', index=False, float_format='%.4f')
    end = time()
    print('Time used: {} min.'.format((end - start) / 60))
