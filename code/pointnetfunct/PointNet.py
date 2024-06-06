import data_process_ml

morpho_path = ".\AneuX\data-v1.0\data\morpho-per-cut.csv"
patient_path = ".\AneuX\data-v1.0\data\clinical.csv"

morpho_data_patient = data_process_ml.read_and_combine_data(morpho_path,patient_path)
merged_dataset = data_process_ml.encode_column(morpho_data_patient)
merged_dataset= data_process_ml.drop_columns(merged_dataset)
merged_dataset= data_process_ml.output_cut1anddome(merged_dataset)