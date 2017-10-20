from data import Data
from model import Model
from time import time
import numpy as np
from stack import Stacker

if __name__ == '__main__':
    # create data class, which import the 2015-2016 crime data in Chicago

    df = Data(2.5, 125)
    # Similar grid search is made for xgboost model. Best parameters are chosen as follows.
    start = time()

    param_rf = {'rf': {'n_estimators': 250,
                       'max_features': 0.25,
                       'min_samples_leaf': 24,
                       'random_state': 1}}
    param_xgb = {'xgb': {'learning_rate': 0.02,
                         'n_estimators': 925,
                         'objective': 'reg:linear',
                         'max_depth': 7,
                         'subsample': 0.575,
                         'colsample_bytree': 0.5,
                         'min_child_weight': 3,
                         'reg_lambda': 0,
                         'reg_alpha': 1,
                         'random_state': 1
                         }}
    param_cat = {'cat': {'iterations': 675,
                         'learning_rate': 0.02,
                         'loss_function': 'MAE',
                         'eval_metric': 'MAE',
                         'l2_leaf_reg': 115,
                         'depth': 6,
                         'random_seed': 1}}
    param_ridge = {'ridge': {'alpha': 0.001}}
    param_ada = {'ada': {'n_estimators': 75,
                         'loss': 'exponential',
                         'learning_rate': 0.1,
                         'random_state': 1}}
    param_lasso = {'lasso': {'alpha': 1}}
    # param_svm = {'svm': {'C': 1e-4}}
    # for i in [75]:
    #     # print(i)
    #     # param_ada['ada']['n_estimators'] = i
    #     model = Model(df, 'svm', param_svm)
    #     # model = Model(df, 'ridge', param_ridge)
    #     model.run_all('mae')

    # for weight in np.linspace(0.1, 1, 5):
    # for i in [1e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]:
    # model = Model(df, 'xgb', param_xgb, 1)
    # print('param: {}'.format(weight))
    # param_rf['rf']['alpha'] = i
    # model = Model(df, 'xgb', param_xgb)
    # model.run_all('mae')

    ridge = Model(df, 'ridge', param_ridge)
    xgb = Model(df, 'xgb', param_xgb)
    rf = Model(df, 'rf', param_rf)
    cat = Model(df, 'cat', param_cat)
    ada = Model(df, 'ada', param_ada)
    lasso = Model(df, 'lasso', param_lasso)
    models = [xgb, rf, cat, ridge, ada, lasso]
    lr_stack = Model(df, 'lr', {'lr': {'fit_intercept': False}})
    # param_xgb = {'xgb': {'learning_rate': 0.02,
    #                      'n_estimators': 1000,
    #                      'objective': 'reg:linear',
    #                      'max_depth': 7,
    #                      # 'subsample': 0.575,
    #                      # 'colsample_bytree': 0.5,
    #                      # 'min_child_weight': 3,
    #                      # 'reg_lambda': 0,
    #                      # 'reg_alpha': 1,
    #                      'random_state': 1
    #                      }}
    # param_cat = {'cat': {'iterations': 675,
    #                      'learning_rate': 0.02,
    #                      'loss_function': 'MAE',
    #                      'eval_metric': 'MAE',
    #                      # 'l2_leaf_reg': 115,
    #                      # 'depth': 6,
    #                      'random_seed': 1}}
    # for i in range(3, 7):
    #     param_xgb['xgb']['max_depth'] = i
    #     xgb_stack = Model(df, 'xgb', param_xgb)
        # rf_stack = Model(df, 'cat', {'rf': {'random_state': 1}})
    stack_models = [lr_stack]
    stack = Stacker(df, 3, models, stack_models, True)
    for j in range(len(stack_models)):
        stack.predict(j, 'mae')
    end = time()
    print('Time used for XGboost: {} min.'.format((end - start) / 60))




