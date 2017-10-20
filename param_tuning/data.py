import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import gc
import os


class Data:
    def __init__(self, outlier_alpha, n_cluster):
        self.target_name = 'logerror'
        self.data = None
        self.target = None
        self.train_num = 0
        self.test_num = 0
        #
        if os.path.exists('../meta_data/data'):
            self.load_data()
        else:
            self.read_data()
            self.preprocess(n_cluster)
            self.split_data(outlier_alpha)
            self.write_data()
        # self.read_data()
        # self.preprocess(n_cluster)
        # self.split_data(outlier_alpha)
        gc.collect()

    def load_data(self):
        """
        Load preprocessed data, if exists
        """
        print('Load data...')
        self.data = pd.read_pickle('../meta_data/data')
        self.target = pd.read_pickle('../meta_data/target')
        self.x_train = pd.read_pickle('../meta_data/xtrain')
        self.y_train = pd.read_pickle('../meta_data/ytrain')
        self.x_test = pd.read_pickle('../meta_data/xtest')
        self.y_test = pd.read_pickle('../meta_data/ytest')
        self.train_num = self.x_train.shape[0]
        self.test_num = self.x_test.shape[0]
        print('\nx_Training set has {} rows, {} columns.'.format(*self.x_train.shape))
        print('x_Test set has {} rows, {} columns.\n'.format(*self.x_test.shape))

    def write_data(self):
        """
        Write preprocessed data
        """
        if not os.path.exists('../meta_data'):
                os.makedirs('../meta_data')
        self.data.to_pickle('../meta_data/data')
        self.target.to_pickle('../meta_data/target')
        self.x_train.to_pickle('../meta_data/xtrain')
        self.y_train.to_pickle('../meta_data/ytrain')
        self.x_test.to_pickle('../meta_data/xtest')
        self.y_test.to_pickle('../meta_data/ytest')
        self.train_num = self.x_train.shape[0]
        self.test_num = self.x_test.shape[0]

    def read_data(self):
        """
        Read in data, concat 2016 and 2017 to self.data
        """
        print('Read in data...')
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
                train_df.to_pickle(train_p)

            if os.path.exists(prop_p):
                prop = pd.read_pickle(prop_p)
            else:
                prop = pd.read_csv(prop_file, low_memory=False)
                print('Binding to float32')
                for col, dtype in zip(prop.columns, prop.dtypes):
                    if dtype == np.float64:
                        prop[col] = prop[col].astype(np.float32)
                prop.to_pickle(prop_p)

            if year == 2016:
                df2016 = train_df.merge(prop, on='parcelid', how='left')
            else:
                df2017 = train_df.merge(prop, on='parcelid', how='left')

        self.data = pd.concat([df2016, df2017], axis=0)
        # to_drop_rf = ['storytypeid', 'fireplaceflag', 'fips'] + ['finishedsquarefeet13', 'architecturalstyletypeid', 'regionidcounty', 'finishedsquarefeet12']\
        #             + ['basementsqrt', 'yardbuildingsqrt26', 'buildingclasstypeid']
        # self.data.drop(to_drop, axis=1, inplace=True)

        del train_df
        del df2016
        del df2017
        del prop
        gc.collect()
        # drop unimportant features with feature importance analysis with xgb, rf, catboost 
        self.target = self.data.logerror

    def fillna_val(self, df, col, val):
        """
        fillna with input val
        """
        df[col].fillna(val, inplace=True)

    def fillna_mean(self, df, col):
        """
        fillna with column mean
        """
        df[col].fillna(df[col].mean(), inplace=True)

    def fillna_neighbor(self, df, col, k):
        """
        fillna with nearest neighbors' value (seems not improving the result)
        """
        model = KNeighborsClassifier(n_neighbors=k)
        train = df.loc[~df[col].isnull(), ['latitude', 'longitude', col]]
        test = df.loc[df[col].isnull(), ['latitude', 'longitude']]
        model.fit(train[['latitude', 'longitude']], train[col])
        df.loc[df[col].isnull(), col] = model.predict(test)

    def encode_label(self, df, col):
        """
        Encode cols of df, the dype of which is object
        """
        le = LabelEncoder()
        le.fit(list(df[col].values))
        df[col] = le.transform(list(df[col].values))

    def log_transform(self, df, col):
        """
        Log transform of input col
        """
        def clean(x):
            if x > 0:
                return x
            elif x == 0:
                return 1e-1
            else:
                return 1e-2

        df[col] = df[col].apply(lambda x: clean(x))
        df[col] = np.log(df[col])

    def create_prop(self, df, col1, col2):
        """
        Create new column by col1/col2
        """
        df[col1 + '_d_' + col2] = df[col1] / df[col2]
        self.fillna_val(df, col1 + '_d_' + col2, 0)
        df[col1 + '_d_' + col2].replace([np.inf, -np.inf], 0, inplace=True)

    def remove_outlier(self, alpha):
        """
        Remove outliers in training data to make the model more robust
        alpha is selected by validation (by convention is 1.5)
        """
        q1 = np.percentile(self.y_train, 25)
        q3 = np.percentile(self.y_train, 75)
        iqr = q3 - q1
        outlier_upper = q3 + alpha * iqr
        outlier_lower = q1 - alpha * iqr
        # print 'Outlier upper bound is {}'.format(outlier_upper)
        # print 'Outlier lower bound is {}'.format(outlier_lower)
        select_index = (self.y_train < outlier_upper) & (self.y_train > outlier_lower)
        self.x_train = self.x_train[select_index]
        self.y_train = self.y_train[select_index]
        # self.data = self.data[(self.target < outlier_upper) & (self.target > outlier_lower)]

    def create_cluster(self, n_cluster):
        """
        Create cluster by Guassian Mixture
        """
        gmm = GaussianMixture(n_components=n_cluster, covariance_type='full', random_state=1)
        gmm.fit(self.data[['latitude', 'longitude']])
        self.data['cluster'] = gmm.predict(self.data[['latitude', 'longitude']])

    def create_cluster_kmeans(self, n_cluster):
        """
        Create cluster by Kmeans
        """
        cluster = KMeans(n_clusters=n_cluster, random_state=10)
        cluster.fit(self.data[['latitude', 'longitude']])
        self.data['cluster'] = cluster.predict(self.data[['latitude', 'longitude']])

    def create_cnt(self, col):
        """
        Create counts (home number) and the logerror std by region
        """
        ct = self.data[col].value_counts().to_dict()
        std = self.data[[col, 'logerror']].groupby(col).agg('std')
        std = dict(std.reset_index().as_matrix())
        self.data[col + '_cnt'] = self.data[col].map(ct)
        self.data[col + '_std'] = self.data[col].map(std)
        self.fillna_val(self.data, col + '_std', 0)

    def create_mean(self, col_to_agg, col_to_group):
        """
        Create col mean by region
        """
        means = self.data[[col_to_agg, col_to_group]].groupby(col_to_group).agg('mean')
        means = dict(means.reset_index().as_matrix())
        self.data[col_to_agg + '_' + col_to_group] = self.data[col_to_group].map(means)

    def create_polar_coor(self):
        """
        Create polar coordinate for the positional data
        """
        center = [self.data['latitude'].mean(), self.data['longitude'].mean()]
        pos = pd.DataFrame()
        pos['x'] = self.data['latitude'] - center[0]
        pos['y'] = self.data['longitude'] - center[1]
        self.data['radius'] = pos['x'] ** 2 + pos['y'] ** 2
        self.data['polar_angle'] = np.arctan2(pos['x'], pos['y'])

    def create_multi(self, col1, col2):
        """
        Create new column by col1 * col2
        """
        self.data[col1 + '_mul_' + col2] = self.data[col1] * self.data[col2]

    def create_add(self, col1, col2):
        """
        Create new column by col1 + col2
        """
        self.data[col1 + '_add_' + col2] = self.data[col1] + self.data[col2]

    def create_minus(self, col1, col2):
        """
        Create new column by col1 - col2
        """
        self.data[col1 + '_m_' + col2] = self.data[col1] + self.data[col2]

    def dummies(self, col, name):
        """
        Create dummies for categorical columns
        """
        series = self.data[col]
        del self.data[col]
        dummies = pd.get_dummies(series, prefix=name)
        self.data = pd.concat([self.data, dummies], axis=1)

    def check_nan_and_infinite(self, df):
        """
        Check nan and infinite value
        """
        m = df.as_matrix()
        if (np.isnan(m).sum()) > 0:
            print('Nan exists:')
            print(np.where(np.isnan(m)))
        if (~np.isfinite(m)).sum() > 0:
            print('Infinite values exist:')
            print(np.where(~np.isfinite(m)))
            print(m[~np.isfinite(m)])

    def preprocess(self, n_cluster):
        """
        Data preprocessing
        """
        print('Preprocessing...')
        # Extrac date info
        self.data['month'] = self.data['transactiondate'].dt.month
        self.data['year'] = self.data['transactiondate'].dt.year

        # Count na value
        self.data['nacnt'] = self.data.isnull().sum(axis=1)

        # In the data description file, airconditioningtypeid 5 corresponds to None
        # In the data description file, airconditioningtypeid 13 corresponds to None
        self.fillna_val(self.data, 'airconditioningtypeid', 5)
        self.fillna_val(self.data, 'heatingorsystemtypeid', 13)

        # For object type column, encode label
        for col in self.data.columns:
            if self.data[col].dtype == 'object':
                self.fillna_val(self.data, col, False)
                self.encode_label(self.data, col)

        # For the col in fill_zero, fillna with zero.
        fill_zero = ['bathroomcnt', 'bedroomcnt', 'buildingqualitytypeid', 'threequarterbathnbr',
                     'finishedfloor1squarefeet', 'calculatedfinishedsquarefeet',
                     'finishedsquarefeet15', 'finishedsquarefeet50', 'fireplacecnt', 'fireplacecnt',
                     'garagecarcnt', 'garagetotalsqft', 'pooltypeid7', 'roomcnt', 'numberofstories', 'poolcnt',
                     'pooltypeid10', 'pooltypeid2', 'unitcnt', 'yardbuildingsqft17', 'fullbathcnt',
                     'poolsizesum', 'finishedsquarefeet6', 'decktypeid', 'buildingclasstypeid',
                     'typeconstructiontypeid', 'yardbuildingsqft26', 'basementsqft', 'calculatedbathnbr'] \
                    + ['storytypeid', 'fireplaceflag', 'fips'] \
                    + ['finishedsquarefeet13', 'architecturalstyletypeid', 'regionidcounty', 'finishedsquarefeet12']
        for col in fill_zero:
            self.fillna_val(self.data, col, 0)

        # For the col in fill_mean, fillna with col mean.
        fill_mean = ['latitude', 'longitude', 'yearbuilt', 'lotsizesquarefeet',
                     'taxvaluedollarcnt', 'structuretaxvaluedollarcnt',
                     'landtaxvaluedollarcnt', 'taxamount', 'assessmentyear',
                     'taxdelinquencyyear', 'propertycountylandusecode', 'propertylandusetypeid',
                     'propertyzoningdesc', 'rawcensustractandblock', 'censustractandblock', 'regionidcity',
                     'regionidzip', 'regionidneighborhood']
        for col in fill_mean:
            self.fillna_mean(self.data, col)

        # Scale large values
        le6 = ['latitude', 'longitude', 'rawcensustractandblock']
        le12 = ['censustractandblock']
        self.data[le6] /= 1e6
        self.data[le12] /= 1e12

        # Create clusters using longitude and latitude
        self.create_cluster(n_cluster)

        # Create selected feature mean by region
        col_agg = ['structuretaxvaluedollarcnt', 'taxamount', 'lotsizesquarefeet', 'calculatedfinishedsquarefeet',
                   'landtaxvaluedollarcnt', 'taxvaluedollarcnt', 'yearbuilt', 'logerror']
        col_group = ['regionidzip', 'regionidcity', 'regionidneighborhood', 'cluster']
        for col1 in col_agg:
            for col2 in col_group:
                self.create_mean(col1, col2)
        self.create_mean('logerror', 'yearbuilt')

        # living area proportions
        self.create_prop(self.data, 'calculatedfinishedsquarefeet', 'lotsizesquarefeet')
        # tax value ratio
        self.create_prop(self.data, 'taxvaluedollarcnt', 'taxamount')
        # tax value ratio2
        self.create_prop(self.data, 'landtaxvaluedollarcnt', 'taxvaluedollarcnt')
        # tax value ratio3
        self.create_prop(self.data, 'taxamount', 'structuretaxvaluedollarcnt')
        # tax value proportions
        self.create_prop(self.data, 'structuretaxvaluedollarcnt', 'landtaxvaluedollarcnt')
        # room ratio
        self.create_prop(self.data, 'bedroomcnt', 'bathroomcnt')
        # living area proportion2, finshedsqurefeet12 are same as calculatedfinishedsquarefeet
        # self.create_prop(self.data, 'finishedsquarefeet12', 'calculatedfinishedsquarefeet')
        # area per room
        self.create_prop(self.data, 'calculatedfinishedsquarefeet', 'roomcnt')
        # room per room
        self.create_prop(self.data, 'roomcnt', 'bedroomcnt')

        # create location related new features
        self.create_multi('latitude', 'longitude')
        self.create_add('latitude', 'longitude')
        self.create_polar_coor()

        # create hount counts and logerror std by region
        create_cnt = ['cluster', 'regionidcity', 'regionidzip', 'regionidneighborhood']
        for col in create_cnt:
            self.create_cnt(col)

        # Log transform
        log_col = ['structuretaxvaluedollarcnt', 'taxamount', 'lotsizesquarefeet', 'censustractandblock',
                   'calculatedfinishedsquarefeet', 'landtaxvaluedollarcnt', 'taxvaluedollarcnt']
        for col in log_col:
            self.log_transform(self.data, col)

        # Extract logerror column to be target
        self.target = self.data[self.target_name]
        self.data = self.data.drop([self.target_name, 'parcelid', 'transactiondate'], axis=1)

        for col, dtype in zip(self.data.columns, self.data.dtypes):
            if dtype == np.float64:
                self.data[col] = self.data[col].astype(np.float32)
            if dtype == np.int64:
                self.data[col] = self.data[col].astype(np.int32)

        self.check_nan_and_infinite(self.data)
        print(self.data.head())

    def split_data(self, outlier_alpha):
        """
        Split data, using only fourth quarter value as test data
        """
        df_n4 = self.data[self.data.month < 10]
        df_n4_t = self.target[self.data.month < 10]
        df_4 = self.data[self.data.month >= 10]
        df_4_t = self.target[self.data.month >= 10]
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(df_4,
                                                                                df_4_t,
                                                                                test_size=0.3,
                                                                                random_state=1)
        self.x_train = pd.concat([df_n4, self.x_train], axis=0)
        self.y_train = pd.concat([df_n4_t, self.y_train], axis=0)

        # Remove outliers
        self.remove_outlier(outlier_alpha)

        self.x_train.reset_index(drop=True, inplace=True)
        self.y_train.reset_index(drop=True, inplace=True)
        self.x_test.reset_index(drop=True, inplace=True)
        self.y_test.reset_index(drop=True, inplace=True)

        self.train_num = self.x_train.shape[0]
        self.test_num = self.x_test.shape[0]
        print('\nx_Training set has {} rows, {} columns.'.format(*self.x_train.shape))
        print('x_Test set has {} rows, {} columns.\n'.format(*self.x_test.shape))

    def data_info(self):
        """
        Info of train and test data
        """
        print('\nTrain:\n{}\n'.format('-' * 50))
        print(self.x_train.info())
        print('\nTrain target:\n{}\n'.format('-' * 50))
        print(self.y_train.info())

