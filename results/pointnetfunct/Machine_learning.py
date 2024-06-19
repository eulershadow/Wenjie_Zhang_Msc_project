import data_process_ml
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import FeatureUnion
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import copy

morpho_path = ".\AneuX\data-v1.0\data\morpho-per-cut.csv"
patient_path = ".\AneuX\data-v1.0\data\clinical.csv"

morpho_data_patient = data_process_ml.read_and_combine_data(morpho_path,patient_path)
merged_dataset = data_process_ml.encode_column(morpho_data_patient)
merged_dataset = data_process_ml.drop_columns(merged_dataset)
morpho_data_cut1,morpho_data_dome = data_process_ml.output_cut1anddome(merged_dataset)

class DataFrameSelector(BaseEstimator):
    
    def __init__(self, attribute_names):
        self.attribute_names= attribute_names
        
    def fit(self,X, y = None):
        return self
    
    def transform(self, X):
        return X[self.attribute_names].values

train_data_cat = copy.deepcopy(morpho_data_cut1)
train_data_cat.drop(["status_ruptured"], axis=1, inplace=True)
data_num = list(train_data_cat)

num_pipeline= Pipeline([
    ('selector', DataFrameSelector(data_num)),
    ('imputer',SimpleImputer(strategy="mean")),
    ('stand_scalar',StandardScaler()),
])


full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline",num_pipeline)
])

target_data_copy = morpho_data_dome["status_ruptured"].copy()

data_copy = morpho_data_dome.copy()
data_test = morpho_data_dome.copy()
data_copy = data_copy[:634]
data_test = data_test[634:]
print(len(data_copy),len(data_test))

train_set, test_set = train_test_split(data_copy , test_size= 0.2, random_state=42)

train_set_target = train_set["status_ruptured"].copy()
train_set.drop(("status_ruptured"),axis=1,inplace=True)

test_set_target = test_set["status_ruptured"].copy()
test_set.drop(("status_ruptured"),axis=1,inplace=True)

data_test_target = data_test["status_ruptured"].copy()
data_test.drop(("status_ruptured"),axis=1,inplace=True)

test_prepared = full_pipeline.fit_transform(test_set)
train_prepared = full_pipeline.fit_transform(train_set)
data_test_prepared = full_pipeline.fit_transform(data_test)