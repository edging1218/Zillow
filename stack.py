import numpy as np
import pandas as pd

class Stacker:
    def __init__(self, features, k_fold, models, stack_models):
        self.models = models
        self.k_fold = k_fold
        self.features = features
        self.stack_models = stack_models
        model_num = len(models)
        self.meta_train = np.zeros((features.train_num, model_num))
        self.meta_test = np.zeros((features.test_num, model_num))
        self.create_meta_feature()
        self.model_stacking()

    def create_meta_feature(self):
        for idx, model in enumerate(self.models):
            ptrain, ptest = model.stacking_feature('mae', self.k_fold)
            self.meta_train[:, [idx]] = ptrain
            self.meta_test[:, [idx]] = ptest
        print('\nMeta_feature created with shape {} in train and {} in test\n'.
              format(self.meta_train.shape, self.meta_test.shape))
        self.meta_train = pd.DataFrame(self.meta_train)
        self.meta_test = pd.DataFrame(self.meta_test)

    def model_stacking(self):
        for model in self.stack_models:
            print('model %s:' % model.model_type)
            model.cross_validation(self.meta_train, self.features.y_train, 'mae', 3)

    def stack_model_grid_search(self, idx):
        model = self.stack_models[idx]
        return model.grid_search(self.meta_train,
                                 self.features.y_train,
                                 'mae',
                                 3,
                                 model.model_param)

    def predict(self, idx):
        model = self.stack_models[idx]
        return model.run(self.meta_train, self.features.y_train,
                         self.meta_test, self.features.y_test,
                         'mae')



