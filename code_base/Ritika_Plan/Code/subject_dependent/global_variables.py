time_window = 30

batch_size = 512

split_ratio = 0.8

data_folder_path = '/home/tusharsingh/code_base/Ritika_Plan/Data/modified_data'
temp_folder_path = "/home/tusharsingh/code_base/Ritika_Plan/Data/Temp"
confusion_matrix_path = '/home/tusharsingh/code_base/Ritika_Plan/Data/conf_channel_specific'
saved_model_path = '/home/tusharsingh/code_base/Ritika_Plan/Data/conf_channel_specific/saved_models'
logging_part = '/home/tusharsingh/code_base/Ritika_Plan/Data/conf_channel_specific/log.txt'

# decided_channels = set(['EEG F3-C3'])

pathology_time_overlap = {
    'BRUX': 0,
    'INS': 0,
    'Normal': 0,
    'NARCO':  0,
    'NFLE': 0,
    'PLM': 0,
    'RBD': 0,
    'SDB': 0
}

channels = {
    # 'brux': ['EEG C3-P3', 'EEG C4-A1', 'EEG C4-P4', 'EEG F3-C3', 'EEG F4-C4', 'EEG F7-T3', 'EEG F8-T4', 'EEG FP1-F3', 'EEG Fp2-F4', 'EEG P3-O1', 'EEG P4-O2', 'EEG T3-T5', 'EEG T4-T6', 'ROC-LOC'],
    'brux': ['ROC-LOC'],
    'ins': ['EEG C3-P3', 'EEG C4-A1', 'EEG C4-P4', 'EEG F3-C3', 'EEG F4-C4', 'EEG F7-T3', 'EEG F8-T4', 'EEG FP1-F3', 'EEG Fp2-F4', 'EEG P3-O1', 'EEG P4-O2', 'ROC-LOC'], 
    'narco':  ['EEG C3-P3', 'EEG C4-A1', 'EEG C4-P4', 'EEG F3-C3', 'EEG F4-C4', 'EEG FP1-F3', 'EEG Fp2-F4', 'EEG P3-O1', 'EEG P4-O2', 'ROC-LOC'], 
    'nfle': ['EEG C3-P3', 'EEG C4-A1', 'EEG C4-P4', 'EEG F3-C3', 'EEG F4-C4', 'EEG F7-T3', 'EEG F8-T4', 'EEG FP1-F3', 'EEG Fp2-F4', 'EEG P3-O1', 'EEG P4-O2', 'EEG T3-T5', 'EEG T4-T6', 'ROC-LOC'],
    'plm': ['EEG C3-P3', 'EEG C4-A1', 'EEG C4-P4', 'EEG F3-C3', 'EEG F4-C4', 'EEG F7-T3', 'EEG F8-T4', 'EEG FP1-F3', 'EEG Fp2-F4', 'EEG P3-O1', 'EEG P4-O2', 'EEG T3-T5', 'EEG T4-T6', 'ROC-LOC'],  
    'rbd': ['EEG C3-P3', 'EEG C4-A1', 'EEG C4-P4', 'EEG F3-C3', 'EEG F4-C4', 'EEG F7-T3', 'EEG F8-T4', 'EEG FP1-F3', 'EEG Fp2-F4', 'EEG P3-O1', 'EEG P4-O2', 'EEG T3-T5', 'EEG T4-T6', 'ROC-LOC'],  
    'sdb': ['EEG C4-A1', 'EEG C4-P4', 'EEG F4-C4', 'EEG Fp2-F4', 'EEG P4-O2', 'ROC-LOC', 'EEG F2-F4']
}



if __name__ == "__main__":
    pass
