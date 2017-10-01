import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split


class Data:
    def __init__(self):
        self.target_name = 'logerror'
        self.data = None
        self.target = None

        self.read_data()
        self.preprocess()
        self.split_data()

    def read_data(self):
        """
        Read in train and test data
        """
        print 'Read in data...'

        train_2016 = pd.read_csv('../input/train_2016_v2.csv', parse_dates=['transactiondate'])
        properties_2016 = pd.read_csv('../input/properties_2016.csv')
        self.train = train_2016.merge(properties_2016, on='parcelid', how='left')

        sample = pd.read_csv('../input/sample_submission.csv')
        sample['parcelid'] = sample['ParcelId']

        self.test = sample.merge(properties_2016, on='parcelid', how='left')
        self.test = self.test.drop(sample.columns.tolist(), axis=1)

        del sample['parcelid']

        self.submission = sample

        print properties_2016.shape
        print train_2016.shape
        print sample.shape
        print self.train.shape
        print self.test.shape

    def fillna_val(self, df, col, val):
        df[col] = df[col].fillna(val)

    def fillna_mean(self, df, col):
        df[col] = df[col].fillna(df[col].mean())

    def fillna_neighbor(self, col, k):
        model = KNeighborsClassifier(n_neighbors=k)
        #    	train = self.train[~self.train[col].isnull()]
        model.fit(self.train[['latitude', 'longitude']], self.train[col])
        for df in [self.train, self.test]:
            if np.any(df[col].isnull()):
                test = df.loc[df[col].isnull(), ['latitude', 'longitude']]
                df.loc[df[col].isnull(), col] = model.predict(test)

    def encode_label(self, df, col):
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    def remove_outlier(self):
        self.target = self.train[self.target_name]
        q1 = np.percentile(self.target, 25)
        q3 = np.percentile(self.target, 75)
        iqr = q3 - q1
        outlier_upper = q3 + 2.5 * iqr
        outlier_lower = q1 - 2.5 * iqr
        print 'Outlier upper bound is {}'.format(outlier_upper)
        print 'Outlier lower bound is {}'.format(outlier_lower)
        self.train = self.train[(self.target < outlier_upper) & (self.target > outlier_lower)]
        self.target = self.train[self.target_name]

    def create_cluster(self):
        gmm = GaussianMixture(n_components=125, covariance_type='full')
        gmm.fit(self.train[['latitude', 'longitude']])
        for df in [self.train, self.test]:
            df.loc[:, 'cluster'] = gmm.predict(df[['latitude', 'longitude']])

    def create_cnt(self, col):
        ct = dict(self.test[col].value_counts())
        newcol = col + 'cnt'

        def extract_ct(x):
            if x in ct:
                return ct[x]
            else:
                return 0

        for df in [self.train, self.test]:
            df.loc[:, newcol] = df[col].apply(lambda x: extract_ct(x))

    def preprocess(self):
        """
        Fill nan value with a value or mean or k nearest neighbor
        remove outliers
        Label encode object type data
        """
        print 'Preprocessing...'

        self.train['month'] = self.train['transactiondate'].dt.month
        datasets = [self.train, self.test]
        for df in datasets:
            df['nacnt'] = df.isnull().sum(axis=1)

            # In the data description file, airconditioningtypeid 5 corresponds to None
            # In the data description file, airconditioningtypeid 13 corresponds to None
            self.fillna_val(df, 'airconditioningtypeid', 5)
            self.fillna_val(df, 'heatingorsystemtypeid', 13)

            # Change object type column
            for col in df.columns:
                if df[col].dtype == 'object':
                    self.fillna_val(df, col, False)
                    self.encode_label(df, col)

            # For the col in fill_zero, fillna with zero.
            fill_zero = ['architecturalstyletypeid', 'basementsqft', 'bathroomcnt',
                         'bedroomcnt', 'buildingqualitytypeid', 'buildingclasstypeid',
                         'calculatedbathnbr', 'decktypeid', 'threequarterbathnbr',
                         'finishedfloor1squarefeet', 'calculatedfinishedsquarefeet',
                         'finishedsquarefeet6', 'finishedsquarefeet12',
                         'finishedsquarefeet13', 'finishedsquarefeet15',
                         'finishedsquarefeet50', 'fireplacecnt', 'fireplacecnt',
                         'garagecarcnt', 'garagetotalsqft',
                         'lotsizesquarefeet', 'numberofstories', 'poolcnt',
                         'poolsizesum', 'pooltypeid10', 'pooltypeid2',
                         'pooltypeid7', 'roomcnt', 'storytypeid',
                         'typeconstructiontypeid', 'unitcnt', 'yardbuildingsqft17',
                         'yardbuildingsqft26', 'fullbathcnt']
            for col in fill_zero:
                self.fillna_val(df, col, 0)

            # For the col in fill_mean, fillna with col mean.
            fill_mean = ['fips', 'latitude', 'longitude', 'yearbuilt',
                         'taxvaluedollarcnt', 'structuretaxvaluedollarcnt',
                         'landtaxvaluedollarcnt', 'taxamount', 'assessmentyear',
                         'taxdelinquencyyear']
            for col in fill_mean:
                self.fillna_mean(df, col)
            # For location related cols, fillna with the vote of 5 nearest neighbors.
            fill_neighbor = ['propertycountylandusecode', 'propertylandusetypeid',
                             'propertyzoningdesc', 'rawcensustractandblock',
                             'censustractandblock', 'regionidcounty', 'regionidcity',
                             'regionidzip', 'regionidneighborhood']
            for col in fill_neighbor:
                # self.fillna_neighbor(col, 5)
                self.fillna_mean(df, col)

            for col, dtype in zip(df.columns, df.dtypes):
                if dtype == np.float64:
                    df[col] = df[col].astype(np.float32)
                if dtype == np.int64:
                    df[col] = df[col].astype(np.int32)

        # Remove outliers
        self.remove_outlier()

        # Cluster the positioning data
        self.create_cluster()

        # House number in each county, city, zip and neighborhood
        create_cnt = ['regionidcounty', 'regionidcity', 'regionidzip', 'regionidneighborhood']
        for col in create_cnt:
            self.create_cnt(col)

        to_drop = [self.target_name, 'parcelid', 'transactiondate']
        self.train = self.train.drop(to_drop, axis=1)
        print self.train.head()

    def assign_test_month(self, month):
        self.test['month'] = month

    def dummies(self, col, name):
        series = self.data[col]
        del self.data[col]
        dummies = pd.get_dummies(series, prefix=name)
        self.data = pd.concat([self.data, dummies], axis=1)

    def split_data(self):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.train,
                                                                                self.target,
                                                                                test_size=0.2,
                                                                                random_state=1)
        print '\nx_Training set has {} rows, {} columns.'.format(*self.x_train.shape)
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

    def write_submission(self, prediction, month):
        """
        Report the final model performance with validation data set
        """
        for col in self.submission.columns[self.submission.columns != 'ParcelId']:
            if col[-2:] == str(month):
                self.submission[col] = prediction


    def submit(self, filename):
        """
        Report the final model performance with validation data set
        """
        print self.submission.head()
        self.submission.to_csv('output/' + filename + '.csv', index=False, float_format='%.4f')
