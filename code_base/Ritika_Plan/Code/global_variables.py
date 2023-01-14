sampling_frequency = 256
highpass = 0.3
lowpass = 30
data_folder_path = "/home/tusharsingh/code_base/Ritika_Plan/Data/cap-sleep-database-1.0.0"
temp_folder_path = "/home/tusharsingh/code_base/Ritika_Plan/Data/Temp"

channels = ['C4A1', 'F4C4', 'C4P4', 'P4O2', 'ROCLOC', 'EMG1EMG2', 'ECG1ECG2', 'SX1SX2', 'Fp2F4', 'HR', 'DX1DX2', 'SAO2', 'PLETH', 'STAT',
                    'F3C3', 'C3P3', 'P3O1', 'FP1F3', 'F8T4', 'F7T3', 'T4T6', 'T3T5', 'TORACE', 'MIC', 'ADDOME', 'Position', 'C3A2', 'FP2F4', 'Pleth', 'Ox Status']

decided_channels = set(['P4O2', 'ROCLOC', 'EMG1EMG2'])

dicided_channels_name = ['P4-O2', 'ROC-LOC', 'EMG1-EMG2']

pathology_dict = {
    # 'brux': 'Bruxism',
    # 'ins': 'Insomnia',
    # 'n': 'control',
    # 'narco':  'Narcolepsy',
    'nfle': 'Nocturnal frontal lobe epilepsy',
    'plm': 'Periodic leg movements',
    'rbd': 'REM behavior disorder',
    # 'sdb': 'Sleep-disordered breathing'
}

# [('C4A1', 102),
#  ('F4C4', 99),
#  ('C4P4', 99),
#  ('P4O2', 99),
#  ('ROCLOC', 96),
#  ('EMG1EMG2', 96),
#  ('ECG1ECG2', 96),
#  ('SX1SX2', 89),
#  ('Fp2F4', 88),
#  ('HR', 88),
#  ('DX1DX2', 86),
#  ('SAO2', 85),
#  ('PLETH', 77),
#  ('STAT', 74),
#  ('F3C3', 70),
#  ('C3P3', 70),
#  ('P3O1', 70),
#  ('FP1F3', 69),
#  ('F8T4', 68),
#  ('F7T3', 68),
#  ('T4T6', 45),
#  ('T3T5', 45),
#  ('TORACE', 31),
#  ('MIC', 27),
#  ('ADDOME', 22),
#  ('Position', 12),
#  ('C3A2', 8),
#  ('FP2F4', 8),
#  ('Pleth', 8),
#  ('Ox Status', 8)]


if __name__ == "__main__":
    pass
