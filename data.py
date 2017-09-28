import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split


class Data:
    def __init__(self, outlier_alpha, n_cluster):
        self.target_name = 'logerror'
        self.data = None
        self.target = None

        self.read_data()
        self.preprocess(n_cluster)
        self.train = self.data
        self.target = self.train[self.target_name]

        # self.train = self.data[self.data['train']]
        self.train = self.train.drop([self.target_name, 'parcelid', 'transactiondate'], axis=1)
        # self.train =  self.train.drop([self.target_name, 'parcelid', 'transactiondate', 'train'], axis=1)
        self.split_data(outlier_alpha)

        # self.test = self.data[~self.data['train']]
        # self.test =  self.test.drop([self.target_name, 'parcelid', 'transactiondate', 'train'], axis=1)

    def read_data(self):
        """
        Read in train and test data
        """
        print 'Read in data...'
        train_2016 = pd.read_csv('input/train_2016_v2.csv', parse_dates=['transactiondate'])
        properties_2016 = pd.read_csv('input/properties_2016.csv')
        self.data = pd.merge(train_2016, properties_2016, on='parcelid', how='left')
        # drop the features with nan value more than 99%
        to_drop = ['poolsizesum', 'finishedsquarefeet6', 'decktypeid', 'buildingclasstypeid', 'finishedsquarefeet13',
                   'typeconstructiontypeid', 'architecturalstyletypeid', 'fireplaceflag', 'yardbuildingsqft26', 'basementsqft',
                   'storytypeid', 'calculatedbathnbr']
        self.data = self.data.drop(to_drop, axis=1)
        self.target = self.data.logerror
        # print self.data.info()

    def fillna_val(self, df, col, val):
        df[col] = df[col].fillna(val)

    def fillna_mean(self, df, col):
        df[col] = df[col].fillna(df[col].mean())

    def fillna_neighbor(self, df, col, k):
        model = KNeighborsClassifier(n_neighbors=k)
        train = df.loc[~df[col].isnull(), ['latitude', 'longitude', col]]
        test = df.loc[df[col].isnull(), ['latitude', 'longitude']]
        model.fit(train[['latitude', 'longitude']], train[col])
        df.loc[df[col].isnull(), col] = model.predict(test)

    def encode_label(self, df, col):
        le = LabelEncoder()
        # df[col] = le.fit_transform(df[col])
        le.fit(list(df[col].values))
        df[col] = le.transform(list(df[col].values))

    def remove_outlier(self, alpha):
        q1 = np.percentile(self.y_train, 25)
        q3 = np.percentile(self.y_train, 75)
        iqr = q3 - q1
        outlier_upper = q3 + alpha * iqr
        outlier_lower = q1 - alpha * iqr
        print 'Outlier upper bound is {}'.format(outlier_upper)
        print 'Outlier lower bound is {}'.format(outlier_lower)
        select_index = (self.y_train < outlier_upper)&(self.y_train > outlier_lower)
        self.x_train = self.x_train[select_index]
        self.y_train = self.y_train[select_index]
        # self.data = self.data[(self.target < outlier_upper) & (self.target > outlier_lower)]

    def create_cluster(self, n_cluster):
        gmm = GaussianMixture(n_components=n_cluster, covariance_type='full')
        gmm.fit(self.data[['latitude', 'longitude']])
        self.data['cluster'] = gmm.predict(self.data[['latitude', 'longitude']])

    def create_cnt(self, col):
        ct = dict(self.data[col].value_counts())
        newcol = col + 'cnt'
        self.data[newcol] = self.data[col].apply(lambda x: ct[x])

    def create_polar_coor(self):
        center = [self.data['latitude'].mean(), self.data['longitude'].mean()]
        pos = pd.DataFrame()
        pos['x'] = self.data['latitude'] - center[0]
        pos['y'] = self.data['longitude'] - center[1]
        self.data['radius'] = pos['x'] ** 2 + pos['y'] ** 2
        self.data['polar_angle'] = np.arctan2(pos['x'], pos['y'])


    def preprocess(self, n_cluster):
        """
        Fill nan value with a value or mean or k nearest neighbor
        remove outliers
        Label encode object type data
        """

        print 'Preprocessing...'
        self.data['month'] = self.data['transactiondate'].dt.month
        self.data['nacnt'] = self.data.isnull().sum(axis=1)



        # In the data description file, airconditioningtypeid 5 corresponds to None
        # In the data description file, airconditioningtypeid 13 corresponds to None
        self.fillna_val(self.data, 'airconditioningtypeid', 5)
        self.fillna_val(self.data, 'heatingorsystemtypeid', 13)

        # Change object type column
        for col in self.data.columns:
            if self.data[col].dtype == 'object':
                self.fillna_val(self.data, col, False)
                self.encode_label(self.data, col)

        # For the col in fill_zero, fillna with zero.
        fill_zero = ['bathroomcnt', 'bedroomcnt', 'buildingqualitytypeid', 'threequarterbathnbr',
                     'finishedfloor1squarefeet', 'calculatedfinishedsquarefeet', 'finishedsquarefeet12',
                     'finishedsquarefeet15', 'finishedsquarefeet50', 'fireplacecnt', 'fireplacecnt',
                     'garagecarcnt', 'garagetotalsqft', 'pooltypeid7', 'roomcnt',
                     'lotsizesquarefeet', 'numberofstories', 'poolcnt', 'pooltypeid10', 'pooltypeid2',
                     'unitcnt', 'yardbuildingsqft17', 'fullbathcnt']
        for col in fill_zero:
            self.fillna_val(self.data, col, 0)

        # For the col in fill_mean, fillna with col mean.
        fill_mean = ['fips', 'latitude', 'longitude', 'yearbuilt',
                     'taxvaluedollarcnt', 'structuretaxvaluedollarcnt',
                     'landtaxvaluedollarcnt', 'taxamount', 'assessmentyear',
                     'taxdelinquencyyear'] + ['propertycountylandusecode', 'propertylandusetypeid',
                         'propertyzoningdesc', 'rawcensustractandblock',
                         'censustractandblock']
        for col in fill_mean:
            self.fillna_mean(self.data, col)

        # For location related cols, fillna with the vote of 5 nearest neighbors.
        fill_neighbor =  ['regionidcounty', 'regionidcity',
                         'regionidzip', 'regionidneighborhood']
        for col in fill_neighbor:
            if self.data[col].isnull().sum() > 0:
                # self.fillna_neighbor(self.data, col, 10)
                self.fillna_mean(self.data, col)

        self.create_cluster(n_cluster)
        create_cnt = ['regionidcounty', 'regionidcity', 'regionidzip', 'regionidneighborhood']
        for col in create_cnt:
            self.create_cnt(col)

        # living area proportions
        self.data['living_area_prop'] = self.data['calculatedfinishedsquarefeet'] / self.data['lotsizesquarefeet']
        # tax value ratio
        self.data['value_ratio'] = self.data['taxvaluedollarcnt'] / self.data['taxamount']
        # tax value proportions
        self.data['value_prop'] = self.data['structuretaxvaluedollarcnt'] / self.data['landtaxvaluedollarcnt']

        for col, dtype in zip(self.data.columns, self.data.dtypes):
            if dtype == np.float64:
                self.data[col] = self.data[col].astype(np.float32)
            if dtype == np.int64:
                self.data[col] = self.data[col].astype(np.int32)

        self.data[['latitude', 'longitude']] /= 1e6
        self.data['censustractandblock'] /= 1e12

        # self.create_polar_coor()

        # print self.data.info()

    def dummies(self, col, name):
        series = self.data[col]
        del self.data[col]
        dummies = pd.get_dummies(series, prefix=name)
        self.data = pd.concat([self.data, dummies], axis=1)

    def split_data(self, outlier_alpha):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.train,
                                                                                self.target,
                                                                                test_size=0.2,
                                                                                random_state=1)
        # Remove outliers
        self.remove_outlier(outlier_alpha)
        print 'x_Training set has {} rows, {} columns.\n'.format(*self.x_train.shape)
        print 'x_Test set has {} rows, {} columns.\n'.format(*self.x_test.shape)

    def data_info(self):
        """
        Info of train and test data
        """
        print '\nTrain:\n{}\n'.format('-' * 50)
        self.x_train.info()
        print '\nTrain target:\n{}\n'.format('-' * 50)
        self.y_train.info()

    def data_peek(self):
        """
        Peek at the train and test data
        """
        print '\nTrain:\n{}\n'.format('-' * 50)
        print self.x_train.head()
        print '\nTrain target:\n{}\n'.format('-' * 50)
        print self.y_train.head()

    def submit(self, prediction):
        """
        Report the final model performance with validation data set
        """
        for col in self.submission.columns:
            if col is not 'ParcelId':
                self.submission[col] = prediction
        self.submission.to_csv('output/xgboost.csv', index=False, float_format='%.4f')
