import matplotlib as plt
import sklearn as sklearn
import scipy as scipy
import pandas as pd
import numpy as np
import copy
from sklearn.preprocessing import OneHotEncoder


morpho_path = ".\AneuX\data-v1.0\data\morpho-per-cut.csv"
patient_path = ".\AneuX\data-v1.0\data\clinical.csv"

def read_and_combine_data(morpho_path,patient_path):
    morpho_data = pd.read_csv(morpho_path)
    patient_data = pd.read_csv(patient_path)

    morpho_data_reform = copy.deepcopy(morpho_data)
    for i in morpho_data_reform:
        str_col = str(i) + "-" + str(morpho_data[i][0]) + "-" + str(morpho_data[i][1])
        #print(str_col)
        morpho_data_reform = morpho_data_reform.rename(columns={i: str_col})
    
    morpho_data_reform = morpho_data_reform.rename(columns={"type-group-index": "source","Unnamed: 1-nan-nan": "dataset","Unnamed: 2-nan-nan": "cuttype"})
    morpho_data_reform = morpho_data_reform.drop([0,1,2])

    morpho_data_patient = morpho_data_reform.merge(patient_data, left_on='dataset', right_on='dataset', how='left')
    morpho_data_patient.drop(['source_y','patientID','vesselFileID','cutToShow',"hospital"], axis=1, inplace=True)
    morpho_data_patient.dropna(subset=['status'], inplace=True)
    morpho_data_patient = morpho_data_patient.reset_index(drop=True)
    
    return morpho_data_patient


def encode_column(morpho_data_patient):
    enc=OneHotEncoder()

    encoded_data = enc.fit_transform(morpho_data_patient[['sex']])
    #print(encoded_data)
    encoded_df = pd.DataFrame(encoded_data.toarray(), columns=enc.get_feature_names_out(['sex']))
    merged_dataset = pd.concat([morpho_data_patient, encoded_df], axis=1)

    encoded_data = enc.fit_transform(merged_dataset[['status']])
    encoded_df = pd.DataFrame(encoded_data.toarray(), columns=enc.get_feature_names_out(['status']))
    merged_dataset = pd.concat([merged_dataset, encoded_df], axis=1)

    # encoded_data = enc.fit_transform(merged_dataset[['location']])
    # encoded_df = pd.DataFrame(encoded_data.toarray(), columns=enc.get_feature_names_out(['location']))
    # merged_dataset = pd.concat([merged_dataset, encoded_df], axis=1)

    # encoded_data = enc.fit_transform(merged_dataset[['side']])
    # encoded_df = pd.DataFrame(encoded_data.toarray(), columns=enc.get_feature_names_out(['side']))
    # merged_dataset = pd.concat([merged_dataset, encoded_df], axis=1)
    
    return merged_dataset

def drop_columns(merged_dataset):
    merged_dataset_copy = copy.deepcopy(merged_dataset)
    merged_dataset_copy.drop(['sex'], axis=1, inplace=True)
    merged_dataset_copy.drop(['status'], axis=1, inplace=True)
    merged_dataset_copy.drop(['status_unruptured'], axis=1, inplace=True)
    merged_dataset_copy.drop(['location'], axis=1, inplace=True)
    merged_dataset_copy.drop(['side'], axis=1, inplace=True)
    
    return merged_dataset_copy

def output_cut1anddome(merged_dataset):
    morpho_data_cut1 = merged_dataset[merged_dataset["cuttype"] == "cut1"]
    morpho_data_cut1.drop(morpho_data_cut1.columns[3:23], axis=1, inplace=True)
    morpho_data_cut1.drop(['source_x',"cuttype","dataset"], axis=1, inplace=True)

    morpho_data_dome = merged_dataset[merged_dataset["cuttype"] == "dome"]
    morpho_data_dome.drop(['source_x',"cuttype","dataset"], axis=1, inplace=True)
    
    return morpho_data_cut1,morpho_data_dome

