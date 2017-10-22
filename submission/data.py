import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import gc
import os


class Data:
    def __init__(self, year):
        self.target_name = 'logerror'
        self.data = None
        self.target = None

        if year != 2016 and year != 2017:
            raise Exception('Invalid year!')
        self.read_data(year)
        self.preprocess()

        self.train_num = self.train.shape[0]
        self.test_num = self.test.shape[0]
        self.total_num = self.train_num + self.test_num

        self.split_data()
        gc.collect()

    def read_data(self, year_test):
        """
        Read in train and test data
        """
        print('Read in data...')

        sample_p = '../input/sample_p'
        if os.path.exists(sample_p):
            sample = pd.read_pickle(sample_p)
        else:
            sample = pd.read_csv('../input/sample_submission.csv')
            # sample.to_pickle(sample_p)
        sample['parcelid'] = sample['ParcelId']
        if year_test == 2016:
            sample.drop(['201710', '201711', '201712'], axis=1, inplace=True)
        elif year_test == 2017:
            sample.drop(['201610', '201611', '201612'], axis=1, inplace=True)

        for year in [2016, 2017]:
            if year == 2016:
                train_p = '../input/train2016'
                prop_p = '../input/prop2016'
                train_file = '../input/train_2016_v2.csv'
                prop_file = '../input/properties_2016.csv'
            elif year == 2017:
                train_p = '../input/train2017'
                prop_p = '../input/prop2017'
                train_file = '../input/train_2017.csv'
                prop_file = '../input/properties_2017.csv'
            if os.path.exists(train_p):
                train_df = pd.read_pickle(train_p)
            else:
                train_df = pd.read_csv(train_file, parse_dates=['transactiondate'])
                # train_df.to_pickle(train_p)

            if os.path.exists(prop_p):
                prop = pd.read_pickle(prop_p)
            else:
                prop = pd.read_csv(prop_file, low_memory=False)
                print('Binding to float32')
                for col, dtype in zip(prop.columns, prop.dtypes):
                    if dtype == np.float64:
                        prop[col] = prop[col].astype(np.float32)
                        # prop.to_pickle(prop_p)
            if year == 2016:
                df2016 = train_df.merge(prop, on='parcelid', how='left')
            else:
                df2017 = train_df.merge(prop, on='parcelid', how='left')
            if year_test == year:
                self.test = sample.merge(prop, on='parcelid', how='left')
                self.test.drop(sample.columns.tolist(), axis=1, inplace=True)
        self.train = pd.concat([df2016, df2017], axis=0)

        del train_df
        del prop
        del sample['parcelid']
        self.submission = sample

        print('File sample shape: {}'.format(sample.shape))
        print('Read_in train set shape: {}'.format(self.train.shape))
        print('Read_in test set shape: {}'.format(self.test.shape))

        # drop unimportant features with feature importance analysis with xgb, rf, catboost 
        to_drop = ['finishedsquarefeet13', 'architecturalstyletypeid', 'fireplaceflag', 'storytypeid', 'regionidcounty',
                   'fips', 'finishedsquarefeet12']
        self.train.drop(to_drop, inplace=True, axis=1)
        self.test.drop(to_drop, inplace=True, axis=1)
        gc.collect()
        return True

    def fillna_val(self, df, col, val):
        """
        fillna with input val
        """
        df[col].fillna(val, inplace=True)

    def fillna_mean(self, col):
        """
        fillna with column mean
        """
        m = self.train[col].mean()
        for df in [self.train, self.test]:
            df[col].fillna(m, inplace=True)

    def encode_label(self, df, col):
        """
        Encode cols of df, the dype of which are object
        """
        le = LabelEncoder()
        le.fit(list(df[col].values))
        df.loc[:, col] = le.transform(list(df[col].values))

    def log_transform(self, df, col):
        """
        Log transform of input col
        """
        # df[col].replace(0, 1e-1, inplace=True)
        # df[col].replace(-1, 1e-2, inplace=True)
        def clean(x):
            if x > 0:
                return x
            elif x == 0:
                return 1e-1
            else:
                return 1e-2
        df.loc[:, col] = df[col].apply(lambda x: clean(x))
        df.loc[:, col] = np.log(df[col])
        # self.fillna_val(df, col, 0)

    def create_prop(self, df, col1, col2):
        """
        Create new column by col1/col2
        """
        df.loc[:, col1 + '_d_' + col2] = df[col1] / df[col2]
        self.fillna_val(df, col1 + '_d_' + col2, 0)
        df.loc[:, col1 + '_d_' + col2].replace([np.inf, -np.inf], 0, inplace=True)

    def create_mult(self, df, col1, col2):
        """
        Create new column by col1 * col2
        """
        df.loc[:, col1 + '_m_' + col2] = df[col1] * df[col2]

    def create_add(self, df, col1, col2):
        """
        Create new column by col1 + col2
        """
        df.loc[:, col1 + '_a_' + col2] = df[col1] + df[col2]

    def remove_outlier(self):
        """
        Remove outliers in training data to make the model more robust
        alpha is selected by validation (by convention is 1.5)
        """
        self.target = self.train[self.target_name]
        q1 = np.percentile(self.target, 25)
        q3 = np.percentile(self.target, 75)
        iqr = q3 - q1
        outlier_upper = q3 + 2.5 * iqr
        outlier_lower = q1 - 2.5 * iqr
        print('Outlier upper bound is {}'.format(outlier_upper))
        print('Outlier lower bound is {}'.format(outlier_lower))
        self.train = self.train.loc[(self.target < outlier_upper) & (self.target > outlier_lower), :]
        self.target = self.train[self.target_name]
        self.train.reset_index(drop=True, inplace=True)
        self.target.reset_index(drop=True, inplace=True)

    def create_cluster(self):
        """
        Create cluster by Guassian Mixture
        """
        gmm = GaussianMixture(n_components=125, covariance_type='full', random_state=1)
        gmm.fit(self.train[['latitude', 'longitude']])
        for df in [self.train, self.test]:
            df.loc[:, 'cluster'] = gmm.predict(df[['latitude', 'longitude']])

    def create_cluster_kmeans(self):
        """
        Create cluster by Kmeans
        """
        cluster = KMeans(n_clusters=50, random_state=10)
        cluster.fit(self.train[['latitude', 'longitude']])
        for df in [self.train, self.test]:
            df.loc[:, 'cluster'] = cluster.predict(df[['latitude', 'longitude']])

    def create_cnt(self, col):
        """
        Create counts (home number) by region
        """
        ct_train = dict(self.train[col].value_counts())

        def extract_ct(x):
            if x in ct_train:
                return ct_train[x]
            else:
                return 0

        for df in [self.train, self.test]:
            df.loc[:, col + '_cnt'] = df[col].apply(lambda x: extract_ct(x))

    def create_logerror_std(self, col):
        """
        Create the logerror std by region
        """
        std_train = dict(self.train[[col, 'logerror']].groupby(col).
                         agg('std').reset_index().as_matrix())

        def extract_val(x, dic):
            if x in dic:
                return dic[x]
            else:
                return 0

        for df in [self.train, self.test]:
            df.loc[:, col + '_std'] = df[col].apply(lambda x: extract_val(x, std_train))
            self.fillna_val(df, col + '_std', 0)

    def create_mean(self, col_to_agg, col_to_group):
        """
        Create col mean by region
        """
        mean_train = dict(self.train[[col_to_agg, col_to_group]].groupby(col_to_group).
                          agg('mean').reset_index().as_matrix())

        def extract_mean(x):
            if x in mean_train:
                return mean_train[x]
            else:
                return 0

        for df in [self.train, self.test]:
            df.loc[:, col_to_agg + '_' + col_to_group] = df[col_to_group].apply(lambda x: extract_mean(x))

    def create_polar_coor(self):
        """
        Create polar coordinate for the positional data
        """
        center = [self.train['latitude'].mean(), self.train['longitude'].mean()]
        for df in [self.train, self.test]:
            pos = pd.DataFrame()
            pos['x'] = df['latitude'] - center[0]
            pos['y'] = df['longitude'] - center[1]
            df.loc[:, 'radius'] = pos['x'] ** 2 + pos['y'] ** 2
            df.loc[:, 'polar_angle'] = np.arctan2(pos['x'], pos['y'])

    def check_nan_and_infinite(self, df):
        """
        Check nan and infinite value
        """
        m = df.as_matrix()
        if np.isnan(m).sum() > 0:
            print('Nan values exist:')
            print(np.where(np.isnan(m)))
        if (~np.isfinite(m)).sum() > 0:
            print('Infinite values exist:')
            print(np.where(~np.isfinite(m)))
            print(m[~np.isfinite(m)])

    def preprocess(self):
        """
        Data preprocessing
        """
        print('Preprocessing...')
        for df in [self.train, self.test]:
            # Count na value in each row
            df.loc[:, 'nacnt'] = df.isnull().sum(axis=1)

            # In the data description file, airconditioningtypeid 5 corresponds to None
            self.fillna_val(df, 'airconditioningtypeid', 5)
            # In the data description file, airconditioningtypeid 13 corresponds to None
            self.fillna_val(df, 'heatingorsystemtypeid', 13)

            # Change object type column
            for col in df.columns:
                if df[col].dtype == 'object':
                    self.fillna_val(df, col, -1)
                    self.encode_label(df, col)

            # For the col in fill_zero, fillna with zero.
            fill_zero = ['bathroomcnt', 'bedroomcnt', 'buildingqualitytypeid', 'threequarterbathnbr',
                         'finishedfloor1squarefeet', 'calculatedfinishedsquarefeet',
                         'finishedsquarefeet15', 'finishedsquarefeet50', 'fireplacecnt', 'fireplacecnt',
                         'garagecarcnt', 'garagetotalsqft', 'pooltypeid7', 'roomcnt', 'numberofstories', 'poolcnt',
                         'pooltypeid10', 'pooltypeid2', 'unitcnt', 'yardbuildingsqft17', 'fullbathcnt',
                         'poolsizesum', 'finishedsquarefeet6', 'decktypeid', 'buildingclasstypeid',
                         'typeconstructiontypeid', 'yardbuildingsqft26',
                         'basementsqft', 'calculatedbathnbr']
            for col in fill_zero:
                self.fillna_val(df, col, 0)

        # For the col in fill_mean, fillna with col mean in train data.
        fill_mean = ['latitude', 'longitude', 'yearbuilt', 'lotsizesquarefeet',
                     'taxvaluedollarcnt', 'structuretaxvaluedollarcnt',
                     'landtaxvaluedollarcnt', 'taxamount', 'assessmentyear',
                     'taxdelinquencyyear', 'propertycountylandusecode', 'propertylandusetypeid',
                     'propertyzoningdesc', 'rawcensustractandblock', 'censustractandblock', 'regionidcity',
                     'regionidzip', 'regionidneighborhood']
        for col in fill_mean:
            self.fillna_mean(col)

        for df in [self.train, self.test]:
            le6 = ['latitude', 'longitude', 'rawcensustractandblock']
            le12 = ['censustractandblock']
            df[le6] /= 1e6
            df[le12] /= 1e12

        # Remove outliers
        self.remove_outlier()

        # Cluster the positioning data
        self.create_cluster()

        col_agg = ['structuretaxvaluedollarcnt', 'taxamount', 'lotsizesquarefeet', 'calculatedfinishedsquarefeet', \
                   'landtaxvaluedollarcnt', 'taxvaluedollarcnt', 'yearbuilt', 'logerror']
        col_group = ['regionidzip', 'regionidcity', 'regionidneighborhood', 'cluster']
        for col1 in col_agg:
            for col2 in col_group:
                self.create_mean(col1, col2)

        self.create_polar_coor()

        # House number in each county, city, zip and neighborhood
        create_cnt = ['cluster', 'regionidcity', 'regionidzip', 'regionidneighborhood']
        for col in create_cnt:
            self.create_cnt(col)
            self.create_logerror_std(col)

        for df in [self.train, self.test]:
            # print(df.shape)
            # living area proportions
            self.create_prop(df, 'calculatedfinishedsquarefeet', 'lotsizesquarefeet')
            # tax value ratio
            self.create_prop(df, 'taxvaluedollarcnt', 'taxamount')
            # tax value ratio2
            self.create_prop(df, 'landtaxvaluedollarcnt', 'taxvaluedollarcnt')
            # tax value ratio3
            self.create_prop(df, 'taxamount', 'structuretaxvaluedollarcnt')
            # tax value proportions
            self.create_prop(df, 'structuretaxvaluedollarcnt', 'landtaxvaluedollarcnt')
            # room ratio
            self.create_prop(df, 'bedroomcnt', 'bathroomcnt')
            # living area proportion2
            # self.create_prop(df, 'finishedsquarefeet12', 'calculatedfinishedsquarefeet')
            # area per room
            self.create_prop(df, 'calculatedfinishedsquarefeet', 'roomcnt')
            # room per room
            self.create_prop(df, 'roomcnt', 'bedroomcnt')

            self.create_mult(df, 'latitude', 'longitude')
            self.create_add(df, 'latitude', 'longitude')
            print(df.shape)

            # Log transform
            log_col = ['structuretaxvaluedollarcnt', 'taxamount', 'lotsizesquarefeet', 'censustractandblock',
                       'calculatedfinishedsquarefeet', 'landtaxvaluedollarcnt', 'taxvaluedollarcnt']
            for col in log_col:
                self.log_transform(df, col)

            for col, dtype in zip(df.columns, df.dtypes):
                if dtype == np.float64:
                    df[col] = df[col].astype(np.float32)
                if dtype == np.int64:
                    df[col] = df[col].astype(np.int32)
            # self.check_nan_and_infinite(df)

        self.train.loc[:, 'year'] = self.train['transactiondate'].dt.year
        self.train.loc[:, 'month'] = self.train['transactiondate'].dt.month

        train_to_drop = [self.target_name, 'parcelid', 'transactiondate']
        self.train.drop(train_to_drop, axis=1, inplace=True)

        print(self.train.head())
        print(self.test.head())
        print('After pre-processing, train set shape: {}'.format(self.train.shape))
        print('After pre-processing, test set shape: {}'.format(self.test.shape))

    def assign_test_month(self, month):
        """
        Assign month in test for prediction
        """
        self.test['month'] = month

    def assign_test_year(self, year):
        """
        Assign month in test for prediction
        """
        self.test['year'] = year

    def dummies(self, col, name):
        """
        Make dummy variable
        """
        series = self.data[col]
        del self.data[col]
        dummies = pd.get_dummies(series, prefix=name)
        self.data = pd.concat([self.data, dummies], axis=1)

    def split_data(self):
        """
        Split data
        """
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.train,
                                                                                self.target,
                                                                                test_size=0.2,
                                                                                random_state=1)
        print('\nx_Training set has {} rows, {} columns.'.format(*self.x_train.shape))
        print('x_Test set has {} rows, {} columns.\n'.format(*self.x_test.shape))

    def data_info(self):
        """
        Info of train and test data
        """
        print('\nTrain:\n{}\n'.format('-' * 50))
        self.x_train.info()
        print('\nTrain target:\n{}\n'.format('-' * 50))
        self.y_train.info()

    def data_peek(self):
        """
        Peek at the train and test data
        """
        print('\nTrain:\n{}\n'.format('-' * 50))
        print(self.x_train.head())
        print('\nTrain target:\n{}\n'.format('-' * 50))
        print(self.y_train.head())

    def write_submission(self, prediction, year, month):
        """
        Report the final model performance with validation data set
        """
        # for col in self.submission.columns[self.submission.columns != 'ParcelId']:
        # if col[-2:] == str(month):
        #     self.submission[col] = prediction
        self.submission[str(year) + str(month)] = prediction
        print(self.submission.head())

    def submit(self, filename, year):
        """
        Report the final model performance with validation data set
        """
        if not os.path.exists('output'):
            os.makedirs('output')
        self.submission.to_csv('output/' + filename + '_' + str(year) + '.csv', index=False, float_format='%.4f')
