import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
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
        print('Read in data...')

        train_2016 = pd.read_csv('../input/train_2016_v2.csv', parse_dates=['transactiondate'])
        properties_2016 = pd.read_csv('../input/properties_2016.csv')
        self.train = train_2016.merge(properties_2016, on='parcelid', how='left')

        sample = pd.read_csv('../input/sample_submission.csv')
        sample['parcelid'] = sample['ParcelId']
        self.test = sample.merge(properties_2016, on='parcelid', how='left')
        self.test = self.test.drop(sample.columns.tolist(), axis=1)

        self.train_num = self.train.shape[0]
        self.test_num = self.test.shape[0]
        self.total_num = self.train_num + self.test_num

        del sample['parcelid']

        self.submission = sample

        print(properties_2016.shape)
        print(train_2016.shape)
        print(sample.shape)
        print(self.train.shape)
        print(self.test.shape)

    def fillna_val(self, df, col, val):
        df[col] = df[col].fillna(val)

    def fillna_mean(self, col):
        m = self.train[col].mean()
        for df in [self.train, self.test]:
            df[col] = df[col].fillna(m)

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
        le.fit(list(df[col].values))
        df[col] = le.transform(list(df[col].values))

    def log_transform(self, df, col):
        df[col].replace(0, 1e-1, inplace=True)
        df[col].replace(-1, 1e-2, inplace=True)
        df[col + '_log'] = np.log(df[col])
        self.fillna_val(df, col, 0)
        # df['col'].replace([np.inf, -np.inf], 0, inplace=True)

    def create_prop(self, df, col1, col2):
        df[col1+'_d_'+col2] = df[col1] / df[col2]
        self.fillna_val(df, col1+'_d_'+col2, 0)
        df[col1+'_d_'+col2].replace([np.inf, -np.inf], 0, inplace=True)

    def remove_outlier(self):
        self.target = self.train[self.target_name]
        q1 = np.percentile(self.target, 25)
        q3 = np.percentile(self.target, 75)
        iqr = q3 - q1
        outlier_upper = q3 + 2.5 * iqr
        outlier_lower = q1 - 2.5 * iqr
        print('Outlier upper bound is {}'.format(outlier_upper))
        print('Outlier lower bound is {}'.format(outlier_lower))
        self.train = self.train[(self.target < outlier_upper) & (self.target > outlier_lower)]
        self.target = self.train[self.target_name]

    def create_cluster(self):
        gmm = GaussianMixture(n_components=125, covariance_type='full')
        gmm.fit(self.train[['latitude', 'longitude']])
        for df in [self.train, self.test]:
            df.loc[:, 'cluster'] = gmm.predict(df[['latitude', 'longitude']])

    def create_cluster_kmeans(self, n_cluster):
        cluster = KMeans(n_clusters=50, random_state=10)
        cluster.fit(self.train[['latitude', 'longitude']])
        for df in [self.train, self.test]:
            df.loc[:, 'cluster'] = cluster.predict(df[['latitude', 'longitude']])

    def create_cnt(self, col):
        ct_train = dict(self.train[col].value_counts())
        ct_test = dict(self.test[col].value_counts())
        newcol = col + 'cnt'

        def extract_ct(x):
            ans = 0
            if x in ct_train:
                ans += ct_train[x]
            if x in ct_test:
                ans += ct_test[x]
            return ans

        for df in [self.train, self.test]:
            df.loc[:, newcol] = df[col].apply(lambda x: extract_ct(x))

    def create_mean(self, col_to_agg, col_to_group):
        mean_train = dict(self.train[[col_to_agg, col_to_group]].groupby(col_to_group).
                          agg('mean').reset_index().as_matrix())
        mean_test = dict(self.test[[col_to_agg, col_to_group]].groupby(col_to_group).
                         agg('mean').reset_index().as_matrix())

        def weighted_mean(x):
            ans = 0
            if x in mean_train:
                ans += mean_train[x] * self.train_num / self.total_num
            if x in mean_test:
                ans += mean_test[x] * self.test_num / self.total_num
            return ans

        for df in [self.train, self.test]:
            df.loc[:, col_to_agg + '_' + col_to_group] = df[col_to_group].apply(lambda x: weighted_mean(x))

    def preprocess(self):
        """
        Fill nan value with a value or mean or k nearest neighbor
        remove outliers
        Label encode object type data
        """
        print('Preprocessing...')
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
                    self.fillna_val(df, col, -1)
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
                     'taxdelinquencyyear'] + ['propertylandusetypeid',
                                              'rawcensustractandblock', 'censustractandblock', 'regionidcounty',
                                              'regionidcity',
                                              'regionidzip', 'regionidneighborhood']
        for col in fill_mean:
            self.fillna_mean(col)

        for df in datasets:
            le6 = ['latitude', 'longitude', 'calculatedfinishedsquarefeet', 'finishedsquarefeet12',
                   'finishedsquarefeet15']
            le8 = ['taxamount', 'lotsizesquarefeet']
            le12 = ['censustractandblock', 'structuretaxvaluedollarcnt', 'taxvaluedollarcnt',
                    'landtaxvaluedollarcnt']
            le17 = ['rawcensustractandblock']
            for col in ['structuretaxvaluedollarcnt', 'taxamount', 'lotsizesquarefeet', 'censustractandblock']:
                self.log_transform(df, col)

            df[le6] /= 1e6
            df[le8] /= 1e8
            df[le12] /= 1e12
            df[le17] /= 1e17

            # living area proportions
            self.create_prop(df, 'calculatedfinishedsquarefeet', 'lotsizesquarefeet')
            # tax value ratio
            self.create_prop(df, 'taxvaluedollarcnt', 'taxamount')
            # tax value proportions
            self.create_prop(df, 'structuretaxvaluedollarcnt', 'landtaxvaluedollarcnt')
            # room ratio
            self.create_prop(df, 'bedroomcnt', 'bathroomcnt')

            for col, dtype in zip(df.columns, df.dtypes):
                if dtype == np.float64:
                    df[col] = df[col].astype(np.float32)
                if dtype == np.int64:
                    df[col] = df[col].astype(np.int32)
                    # print df['latitude'].describe()
                    # print df['longitude'].describe()

        # Remove outliers
        self.remove_outlier()

        # Cluster the positioning data
        self.create_cluster_kmeans()


        col_agg = ['structuretaxvaluedollarcnt', 'taxamount', 'lotsizesquarefeet', 'calculatedfinishedsquarefeet',
                   'finishedsquarefeet12', 'yearbuilt']
        col_group = ['regionidzip', 'regionidcity', 'regionidneighborhood', 'cluster']

        for col1 in col_agg:
            for col2 in col_group:
                self.create_mean(col1, col2)

        # House number in each county, city, zip and neighborhood
        create_cnt = ['regionidcounty', 'regionidcity', 'regionidzip', 'regionidneighborhood']
        for col in create_cnt:
            self.create_cnt(col)

        self.train['month'] = self.train['transactiondate'].dt.month
        to_drop = [self.target_name, 'parcelid', 'transactiondate']
        self.train = self.train.drop(to_drop, axis=1)

        print(self.train.head())
        # print(self.train.columns[60:63])
        # m = self.train.as_matrix()
        # print(m[~np.isfinite(m)])
        # print(np.where(~np.isfinite(m)))
        # m = self.test.as_matrix()
        # print(m[~np.isfinite(m)])
        # print(np.where(~np.isfinite(m)))

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
        print(self.submission.head())
        self.submission.to_csv('output/' + filename + '.csv', index=False, float_format='%.4f')
