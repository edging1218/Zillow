from data import Data
from model import Model
from time import time
import numpy as np
from stack import Stacker

if __name__ == '__main__':
    # import and preprocess data
    df = Data(2.5, 125)
    # Record start time
    start = time()

    # parameter tuned for different model
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

    # Initialize models
    ridge = Model(df, 'ridge', param_ridge)
    xgb = Model(df, 'xgb', param_xgb)
    rf = Model(df, 'rf', param_rf)
    cat = Model(df, 'cat', param_cat)
    ada = Model(df, 'ada', param_ada)
    lasso = Model(df, 'lasso', param_lasso)
    models = [xgb, rf, cat, ridge, ada, lasso]

    # Use simple linear regression and xgb for second level stacking model.
    lr_stack = Model(df, 'lr', {'lr': {'fit_intercept': False}})
    xbg_stack = Model(df, 'xgb', {'xgb': {'random_state': 2}})
    stack_models = [lr_stack, xgb_stack]

    # Create stacker
    stack = Stacker(df, 3, models, stack_models, True)

    # Examine the performance
    for j in range(len(stack_models)):
        stack.predict(j, 'mae')
    end = time()
    print('Time used for XGboost: {} min.'.format((end - start) / 60))




