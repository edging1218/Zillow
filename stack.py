import numpy as np
import pandas as pd

class Stacker:
    def __init__(self, features, k_fold, models, stack_models, file=False):
        self.models = models
        self.k_fold = k_fold
        self.features = features
        self.stack_models = stack_models
        self.model_num = len(models)
        self.stack_model_num = len(stack_models)
        self.meta_train = np.zeros((features.train_num, self.model_num))
        self.meta_test = np.zeros((features.test_num, self.model_num))
        if file:
            self.read_meta_feature()
        else:
            self.create_meta_feature()

    def read_meta_feature(self):
        self.meta_train = pd.read_csv('meta_features/train.csv')
        self.meta_test = pd.read_csv('meta_features/test.csv')

    def create_meta_feature(self):
        for idx, model in enumerate(self.models):
            ptrain, ptest = model.stacking_feature('mae', self.k_fold)
            self.meta_train[:, [idx]] = ptrain
            self.meta_test[:, [idx]] = ptest
        print('\nMeta_feature created with shape {} in train and {} in test\n'.
              format(self.meta_train.shape, self.meta_test.shape))
        self.meta_train = pd.DataFrame(self.meta_train)
        self.meta_test = pd.DataFrame(self.meta_test)
        self.meta_train.to_csv('meta_features/train.csv', index=False)
        self.meta_test.to_csv('meta_features/test.csv', index=False)

    def stack_model_cv(self, metrics, k_fold):
        for model in self.stack_models:
            print('model %s:' % model.model_type)
            model.cross_validation(self.meta_train, self.features.y_train, metrics, k_fold)

    def stack_model_grid_search(self, idx, metrics, k_fold):
        if (idx + 1) > self.stack_model_num or idx < 0:
            print('Invalid model id input.')
            return False
        model = self.stack_models[idx]
        return model.grid_search(self.meta_train,
                                 self.features.y_train,
                                 metrics,
                                 k_fold,
                                 model.model_param)

    # def weighted_stack_model(self, weights):


    def predict(self, idx, metrics):
        if (idx + 1) > self.stack_model_num or idx < 0:
            print('Invalid model id input.')
            return False
        model = self.stack_models[idx]
        # print('Predict for model: {}'.format(model.get_model_name()))
        return model.run(self.meta_train, self.features.y_train,
                         self.meta_test, self.features.y_test,
                         metrics)


