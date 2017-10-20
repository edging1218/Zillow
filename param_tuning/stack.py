import numpy as np
import pandas as pd
import os

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
        """
        Read in precreated metafeatures
        """
        self.meta_train = pd.read_pickle('meta_features/train')
        self.meta_test = pd.read_pickle('meta_features/test')
        print(self.meta_train.head())

    def create_meta_feature(self):
        """
        Create metafeatures and save 
        """
        for idx, model in enumerate(self.models):
            ptrain, ptest = model.stacking_feature('mae', self.k_fold)
            self.meta_train[:, [idx]] = ptrain
            self.meta_test[:, [idx]] = ptest
        print('\nMeta_feature created with shape {} in train and {} in test\n'.
              format(self.meta_train.shape, self.meta_test.shape))
        self.meta_train = pd.DataFrame(self.meta_train)
        self.meta_test = pd.DataFrame(self.meta_test)
        if not os.path.exists('meta_features'):
            os.makedirs('meta_features')
        self.meta_train.to_pickle('meta_features/train')
        self.meta_test.to_pickle('meta_features/test')

    def stack_model_cv(self, metrics, k_fold):
        """
        Stack model cross validation on training data
        """
        for model in self.stack_models:
            print('model %s:' % model.model_type)
            model.cross_validation(self.meta_train, self.features.y_train, metrics, k_fold)

    def stack_model_grid_search(self, idx, metrics, k_fold):
        """
        Stack model parameter grid search by training data
        """
        if (idx + 1) > self.stack_model_num or idx < 0:
            print('Invalid model id input.')
            return False
        model = self.stack_models[idx]
        return model.grid_search(self.meta_train,
                                 self.features.y_train,
                                 metrics,
                                 k_fold,
                                 model.model_param)

    def predict(self, idx, metrics):
        """
        Stack model performance on test data
        """
        if (idx + 1) > self.stack_model_num or idx < 0:
            print('Invalid model id input.')
            return False
        model = self.stack_models[idx]
        print('Predict for model: {}'.format(model.model_type))
        return model.run(self.meta_train, self.features.y_train,
                         self.meta_test, self.features.y_test,
                         metrics)


