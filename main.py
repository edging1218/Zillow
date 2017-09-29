from data import Data
from model import Model
from time import time

if __name__ == '__main__':
    # create data class, which import the 2015-2016 crime data in Chicago

    df = Data(2.5, 125)
    # Similar grid search is made for xgboost model. Best parameters are chosen as follows.
    # start = time()
    # param_xgb = {'xgb_grid': {'learning_rate': [0.02],
    #                           'n_estimators': [550],
    #                           'objective': ['reg:linear'],
    #                           'max_depth': [5],
    #                           'gamma': [0],
    #                           'subsample': [0.8, 0.75, 0.7, 0.65, 0.6],
    #                           'min_child_weight': [2]
    #                            }}
    param_xgb = {'xgb': {'learning_rate': 0.02,
                         'n_estimators': 550,
                         'objective': 'reg:linear',
                         'max_depth': 5,
                         'gamma': 0,
                         'subsample': 0.75,
                         'min_child_weight': 5
                         }}

    # param_xgb = {'xgb': {'eta': 0.037,
    #                      'max_depth': 5,
    #                      'subsample': 0.80,
    #                      'objective': 'reg:linear',
    #                      'lambda': 0.8,
    #                      'alpha': 0.4,
    #                      }}
    xgb = Model(df, 'xgb', param_xgb)
    # xgb.grid_search_all('neg_mean_absolute_error', 3)
    xgb.run_all('mae')
    # end = time()
    # print 'Time used for XGboost: {} min.'.format((end - start) / 60)
