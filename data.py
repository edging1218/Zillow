import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

class Data:
    def __init__(self):
        self.target_name = 'logerror'
        self.data = None
	self.target = None

        self.read_data()
	self.train = self.data
	self.target = self.train[self.target_name]

        self.preprocess()

#	self.train = self.data[self.data['train']]
        self.train =  self.train.drop([self.target_name, 'parcelid', 'transactiondate'], axis=1)
#       self.train =  self.train.drop([self.target_name, 'parcelid', 'transactiondate', 'train'], axis=1)
        self.split_data()

#	self.test = self.data[~self.data['train']]
#       self.test =  self.test.drop([self.target_name, 'parcelid', 'transactiondate', 'train'], axis=1)

    def read_data(self):
        """
        Read in train and test data
        """
        print 'Read in data...'
	train_2016 = pd.read_csv('input/train_2016_v2.csv', parse_dates = ['transactiondate'])
#	train_2016['train'] = True

	properties_2016 = pd.read_csv('input/properties_2016.csv')
	self.data = pd.merge(train_2016, properties_2016, on='parcelid', how='left')
#	sample = pd.read_csv('input/sample_submission.csv')
	# sample.rename(columns = {'ParcelId': 'parcelid'}, inplace=True)
#	sample['parcelid'] = sample['ParcelId']

#	all_data = pd.merge(train_2016, properties_2016, on='parcelid', how='outer')

#	self.data = pd.merge(sample, all_data, on='parcelid', how='left')
#	del sample['parcelid']

#	self.submission = sample
	print self.data.info()

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
	df[col] = le.fit_transform(df[col])

    def remove_outlier(self):
	q1 = np.percentile(self.target, 25)
	q3 = np.percentile(self.target, 75)
	iqr = q3 - q1
	outlier_upper = q3 + 1.5 * iqr
	outlier_lower = q1 - 1.5 * iqr
	print 'Outlier upper bound is {}'.format(outlier_upper)
	print 'Outlier lower bound is {}'.format(outlier_lower)
	# select_index = (self.y_train < outlier_upper)&(self.y_train > outlier_lower)
	# self.x_train = self.x_train[select_index]
	# self.y_train = self.y_train[select_index]
	self.data = self.data[(self.target < outlier_upper) & (self.target > outlier_lower)]

    def preprocess(self):
	'''
	Fill nan value with a value or mean or k nearest neighbor
	remove outliers
	Label encode object type data
	'''
	print 'Preprocessing...'
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
	fill_zero = ['architecturalstyletypeid','basementsqft', 'bathroomcnt',
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
		self.fillna_val(self.data, col, 0)

	# For the col in fill_mean, fillna with col mean. 
	fill_mean = ['fips', 'latitude', 'longitude', 'yearbuilt', 
	'taxvaluedollarcnt', 'structuretaxvaluedollarcnt',
	'landtaxvaluedollarcnt', 'taxamount', 'assessmentyear',
	'taxdelinquencyyear']
	for col in fill_mean:
		self.fillna_mean(self.data, col)

	# For location related cols, fillna with the vote of 5 nearest neighbors. 
	fill_neighbor = ['propertycountylandusecode', 'propertylandusetypeid',
	'propertyzoningdesc', 'rawcensustractandblock', 
	'censustractandblock', 'regionidcounty', 'regionidcity',
	'regionidzip', 'regionidneighborhood']
	for col in fill_neighbor:
		if self.data[col].isnull().sum() > 0:
			self.fillna_neighbor(self.data, col, 5)

	# Remove outliers
	self.remove_outlier()

	print self.data.info()


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
        print '\nTrain:\n{}\n'.format('-'*50)
        print self.x_train.head()
        print '\nTrain target:\n{}\n'.format('-'*50)
        print self.y_train.head()

    def submit(self, prediction):
        """
        Report the final model performance with validation data set
        """
	for col in self.submission.columns:
	    if col is not 'ParcelId':
		self.submission[col] = prediction
	self.submission.to_csv('output/xgboost.csv', index=False, float_format='%.4f')

