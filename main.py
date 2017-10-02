from data import Data
from model import Model
from time import time
from stack import Stacker

if __name__ == '__main__':
    # create data class, which import the 2015-2016 crime data in Chicago

    df = Data(2.5, 125)
    # Similar grid search is made for xgboost model. Best parameters are chosen as follows.
    start = time()
    # param_xgb = {'xgb_grid': {'learning_rate': [0.02],
    #                           'n_estimators': [600],
    #                           'objective': ['reg:linear'],
    #                           'max_depth': [5],
    #                           'gamma': [0],
    #                           'subsample': [0.55],
    #                           'min_child_weight': [4],
    #                           'colsample_bytree': [1],
    # #                         'reg_alpha': [0.5, 1, 1.5],
    #                           'reg_lambda': [50, 100, 500]
    #                            }}
    param_xgb = {'xgb': {'learning_rate': 0.02,
                         'n_estimators': 600,
                         'objective': 'reg:linear',
                         'max_depth': 5,
                         'subsample': 0.55,
                         'min_child_weight': 4,
                         'random_state': 1,
                         'reg_lambda': 50
                         }}
    xgb = Model(df, 'xgb', param_xgb)
    lasso = Model(df, 'lasso', {'lasso': {'alpha': 18}})
    ridge = Model(df, 'ridge', {'ridge': {'alpha': 0.5}})
    rf = Model(df, 'rf', {'rf': {'n_estimators': 500,
                                 'max_features': 'sqrt',
                                 'min_samples_leaf': 35,
                                 'random_state': 1}})
    extra = Model(df, 'extra', {'extra': {'n_estimators': 300, 'random_state': 1}})
    svm = Model(df, 'svm', {'svm': {'C': 1e-4}})
    knn = Model(df, 'knn', {'knn': {'n_neighbors': 64}})
    ada = Model(df, 'adaboost', {'adaboost': {'n_estimators': 400,
                                              'loss': 'exponential',
                                              'learning_rate': 0.01,
                                              'random_state': 1}})
    models = [lasso, ridge, ada, rf, xgb, svm, knn, extra]

    ridge_stack = Model(df, 'ridge', {'ridge_grid': {'alpha': [0.0025, 0.005, 0.0075]}})
    stack_param_xgb = {'xgb_grid': {'learning_rate': [0.02],
                                    'n_estimators': [600],
                                    'objective': ['reg:linear'],
                                    'max_depth': [3, 4],
                                    'gamma': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
                                    'subsample': [1],
                                    'min_child_weight': [1],
                                    'random_state': [1]
                                    }}
    xgb_stack = Model(df, 'xgb', stack_param_xgb)
    rf_stack = Model(df, 'rf', {'rf_grid': {'n_estimators': [100, 200, 300, 400, 500],
                                            'max_features': ['auto', 'sqrt', 'log2'],
                                            'min_samples_leaf': [1, 10, 30, 50]}})
    stack_models = [ridge_stack, xgb_stack, rf_stack]

    stack = Stacker(df, 3, models, stack_models, True)
    for i in range(len(stack_models)):
        stack.stack_model_grid_search(i, 'neg_mean_absolute_error', 3)
        stack.predict(i, 'mae')

    # model.grid_search_all('neg_mean_absolute_error', 3)
    # # model.cross_validation_all('mae', 3)
    # model.run_all('mae')

    end = time()
    print('Time used for XGboost: {} min.'.format((end - start) / 60))
