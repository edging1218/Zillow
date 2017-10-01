from data import Data
from model import Model
from time import time
from stack import Stacker

if __name__ == '__main__':
    # create data class, which import the 2015-2016 crime data in Chicago

    df = Data(2.5, 50)
    # Similar grid search is made for xgboost model. Best parameters are chosen as follows.
    start = time()
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

    # lasso = Model(df, 'lasso', {'lasso': {'alpha':  18}})
    # ridge = Model(df, 'ridge', {'ridge': {'alpha': 0.5}})
    # lasso_stack = Model(df, 'ridge', {'ridge': {}})
    # stack = Stacker(df, 3, [lasso, ridge], [lasso_stack])
    # stack.predict(0)

    # model = Model(df, 'rf', {'rf_grid': {'n_estimators': [100], 'max_features': ['sqrt', 'log2', 'auto'], 'min_samples_leaf':[10, 20, 30, 40, 50]}})
    # model.grid_search_all('neg_mean_absolute_error', 3)
    # model.run_all('mae')

    # model.run_all('mae')
    # model = Model(df, 'svm', {'svm_grid': {'C': [0.5, 1]}})
    # model.grid_search_all('neg_mean_absolute_error', 3)
    # model.run_all('mae')

    # model = Model(df, 'mlp', {'mlp_grid': {'alpha': [1e-4], 'hidden_layer_sizes': (256, 64, 16,), 'learning_rate_init': [0.0001], 'solver':['lbfgs'], 'random_state':[1]}})
    model = Model(df, 'svm', {'svm_grid': {'C':  [0.000001, 0.00001]}})
    model.grid_search_all('neg_mean_absolute_error', 3)
    model.run_all('mae')


    end = time()
    print('Time used for XGboost: {} min.'.format((end - start) / 60))
