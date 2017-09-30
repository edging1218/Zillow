from data import Data
from model import Model
from time import time

if __name__ == '__main__':
    # create data class, which import the 2015-2016 crime data in Chicago

    df = Data(2.5, 50)
    # Similar grid search is made for xgboost model. Best parameters are chosen as follows.
    # start = time()
    # param_xgb = {'xgb_grid': {'learning_rate': [0.02],
    #                           'n_estimators': [550, 575],
    #                           'objective': ['reg:linear'],
    #                           'max_depth': [5],
    #                           'gamma': [0, 0.05],
    #                           'subsample': [0.75, 0.7, 0.65],
    #                           'min_child_weight': [2, 3]
    #                            }}

    # param_xgb = {'xgb': {'learning_rate': 0.02,
    #                      'n_estimators': 550,
    #                      'objective': 'reg:linear',
    #                      'max_depth': 5,
    #                      'gamma': 0,
    #                      'subsample': 0.75,
    #                      'min_child_weight': 2
    #                      }}

    # xgb = Model(df, 'xgb', param_xgb)
    # xgb.grid_search_all('neg_mean_absolute_error', 3)
    # xgb.run_all('mae')

    extra = Model(df, 'extra', {'extra': {}})
    extra.run_all('mae')


    # end = time()
    # print 'Time used for XGboost: {} min.'.format((end - start) / 60)
