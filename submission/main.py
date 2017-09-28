from data import Data
from model import Model
from time import time

if __name__ == '__main__':
    # create data class, which import the 2015-2016 crime data in Chicago
    df = Data()
    # create a model class with grid search parameters by accuracy
    # Optimal parameters are assigned
    # run_all function fits the model then evaluates the performance in test data
    # start = time()
    # param_logit_grid = {'logit_grid':
    #                        {'penalty': ['l1', 'l2'],
    #                         'C': [10 ** i for i in range(-3, 1, 1)]}}
    # logit = Model(crimes, 'logit', param_logit_grid)
    # logit.grid_search_all('accuracy', 3)
    # logit.run_all('accuracy', 3)
    # end = time()
    # print 'Time used for logistic regression: {} min.'.format((end - start) / 60)

    # Similar grid search is made for xgboost model. Best parameters are chosen as follows.
    param_xgb = {'xgb': {'learning_rate': 0.02,
                         'n_estimators': 550,
                         'objective': 'reg:linear',
                         'max_depth': 5
                         }}
    for month in [10, 11, 12]:
        df.assign_test_month(month)
        xgb = Model(df, 'xgb', param_xgb)
        pred = xgb.run_submission()
        df.write_submission(pred, month)

    df.submit(pred, 'xgboost_new_feature')
