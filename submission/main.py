from data import Data
from model import Model
from time import time
import pandas as pd
import os
from stack import Stacker

if __name__ == '__main__':
    start = time()

    # df.assign_test_month(10)
    # lasso = Model(df, 'lasso', {'lasso': {'alpha':  18}})
    # ridge = Model(df, 'ridge', {'ridge': {'alpha': 0.5}})
    # ridge = Model(df, 'ridge', {'ridge': {}})
    # stack = Stacker(df, 3, [lasso, ridge], [lasso_stack])
    # stack.predict(0)
    #
    # param_xgb = {'xgb': {'learning_rate': 0.02,
    #                      'n_estimators': 925,
    #                      'objective': 'reg:linear',
    #                      'max_depth': 7,
    #                      'subsample': 0.575,
    #                      'colsample_bytree': 0.5,
    #                      'min_child_weight': 3,
    #                      'reg_lambda': 0,
    #                      'reg_alpha': 1,
    #                      'random_state': 30
    #                      }}
    # param_cat = {'cat': {'iterations': 675,
    #                      'learning_rate': 0.02,
    #                      'loss_function': 'MAE',
    #                      'eval_metric': 'MAE',
    #                      'l2_leaf_reg': 115,
    #                      'depth': 6,
    #                      'random_seed': 20}}
    # param_rf = {'rf': {'n_estimators': 250,
    #                    'max_features': 0.25,
    #                    'min_samples_leaf': 24,
    #                    'random_state': 1}}
    param_ridge = {'ridge': {'alpha': 0.001}}
    filename = 'ridge_w1'
    for year in [2016, 2017]:
        df = Data(year)
        df.assign_test_year(year)
        model = Model(df, 'ridge', param_ridge)
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
