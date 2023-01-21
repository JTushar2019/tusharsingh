sampling_frequency = 256
highpass = 0.3
lowpass = 30

time_window = 30

batch_size = 512

split_ratio = 0.8

data_folder_path = '/home/tusharsingh/code_base/Ritika_Plan/Data/modified_data'
temp_folder_path = "/home/tusharsingh/code_base/Ritika_Plan/Data/Temp"
confusion_matrix_path = '/home/tusharsingh/code_base/Ritika_Plan/Data/confusion_matrices'
saved_model_path = '/home/tusharsingh/code_base/Ritika_Plan/Data/saved_models'

channels = ['C4A1', 'F4C4', 'C4P4', 'P4O2', 'ROCLOC', 'EMG1EMG2', 'ECG1ECG2', 'SX1SX2', 'Fp2F4', 'HR', 'DX1DX2', 'SAO2', 'PLETH', 'STAT',
                    'F3C3', 'C3P3', 'P3O1', 'FP1F3', 'F8T4', 'F7T3', 'T4T6', 'T3T5', 'TORACE', 'MIC', 'ADDOME', 'Position', 'C3A2', 'FP2F4', 'Pleth', 'Ox Status']

decided_channels = set(['EEG P4-O2', 'ROC-LOC'])

dicided_channels_name = ['EEG P4-O2',  'ROC-LOC']

pathology_dict = {
    # 'brux': 'Bruxism',
    'ins': 'Insomnia',
    'n': 'Control',
    'narco':  'Narcolepsy',
    'nfle': 'Nocturnal frontal lobe epilepsy',
    'plm': 'Periodic leg movements',
    'rbd': 'REM behavior disorder',
    'sdb': 'Sleep-disordered breathing'
}

pathology_time_overlap = {
    'Bruxism': 0,
    'Insomnia': 0,
    'Control': 0,
    'Narcolepsy':  0,
    'Nocturnal frontal lobe epilepsy': 0,
    'Periodic leg movements': 0,
    'REM behavior disorder': 0,
    'Sleep-disordered breathing': 0
}



if __name__ == "__main__":
    pass
