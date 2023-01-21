sampling_frequency = 256
highpass = 0.3
lowpass = 30

time_window = 30
batch_size = 512
split_ratio = 0.8

data_folder_path = '/home/tusharsingh/code_base/Ritika_Plan/Data/modified_data'
temp_folder_path = "/home/tusharsingh/code_base/Ritika_Plan/Data/Temp"
working_data_path = '/home/tusharsingh/code_base/Ritika_Plan/Data/subject_dependent_results'
confusion_matrix_path = '/home/tusharsingh/code_base/Ritika_Plan/Data/subject_dependent_results/graphs'
saved_model_path = '/home/tusharsingh/code_base/Ritika_Plan/Data/subject_dependent_results/saved_models'
tracking_file_path = '/home/tusharsingh/code_base/Ritika_Plan/Data/subject_dependent_results/results.txt'

channels = ['C4A1', 'F4C4', 'C4P4', 'P4O2', 'ROCLOC', 'EMG1EMG2', 'ECG1ECG2', 'SX1SX2', 'Fp2F4', 'HR', 'DX1DX2', 'SAO2', 'PLETH', 'STAT',
                    'F3C3', 'C3P3', 'P3O1', 'FP1F3', 'F8T4', 'F7T3', 'T4T6', 'T3T5', 'TORACE', 'MIC', 'ADDOME', 'Position', 'C3A2', 'FP2F4', 'Pleth', 'Ox Status']

decided_channels = set(['EEG P4-O2', 'EEG C4-P4', 'ROC-LOC'])

pathology_dict = {
    # 'brux': 'Bruxism',
    # 'ins': 'Insomnia',                           #done
    'n': 'Control',
    # 'narco':  'Narcolepsy',                      #done
    'nfle': 'Nocturnal frontal lobe epilepsy',  # done
    # 'plm': 'Periodic leg movements',             #done
    # 'rbd': 'REM behavior disorder',              #done
    # 'sdb': 'Sleep-disordered breathing'
}

train_over_sampling = True
weighted_cross_entropy = False
weighted_random_sampling = False



