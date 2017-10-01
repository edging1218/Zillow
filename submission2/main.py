from data import Data
from model import Model
from time import time
from stack import Stacker

if __name__ == '__main__':
    # create data class, which import the 2015-2016 crime data in Chicago
    df = Data()
    # create a model class with grid search parameters by accuracy
    # Optimal parameters are assigned
    # run_all function fits the model then evaluates the performance in test data
    # Similar grid search is made for xgboost model. Best parameters are chosen as follows.
    param_xgb = {'xgb': {'learning_rate': 0.02,
                          'n_estimators': 550,
                          'objective': 'reg:linear',
                          'max_depth': 5,
                          'subsample': 0.7,
                          'min_child_weight': 3
                          }}
    #
    # lasso = Model(df, 'lasso', {'lasso': {'alpha':  18}})
    # ridge = Model(df, 'ridge', {'ridge': {'alpha': 0.5}})
    # lasso_stack = Model(df, 'ridge', {'ridge': {}})
    # stack = Stacker(df, 3, [lasso, ridge], [lasso_stack])
    # stack.predict(0)

    start = time()
    for month in [10, 11, 12]:
        df.assign_test_month(month)
        #model = Model(df, 'lasso', {'lasso': {'alpha': 20}})
        model = Model(df, 'xgb', param_xgb)
        pred = model.run_submission()
        df.write_submission(pred, month)
    
    df.submit('xgboost_kmeans')
    end = time()
    print('Time used: {} min.'.format((end - start) / 60))
