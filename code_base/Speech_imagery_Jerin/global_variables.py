folder_path = {"Long_words": "/home/tusharsingh/DATAs/speech_EEG/Long_words",
               "Short_Long_words": "/home/tusharsingh/DATAs/speech_EEG/Short_Long_words",
               "Short_words": "/home/tusharsingh/DATAs/speech_EEG/Short_words",
               "Vowels": "/home/tusharsingh/DATAs/speech_EEG/Vowels"}

words_dict = {
    "Long_words": ["cooperate", "independent"],
    "Short_Long_words": ["cooperate", "in"],
    "Short_words": ["out", "in", "up"],
    "Vowels": ["a", "i", "u"]
}

numeric_labels = {
    "Long_words": {"cooperate": 0, "independent": 1},
    "Short_Long_words": {"cooperate": 0, "in": 1},
    "Short_words": {"out": 0, "in": 1, "up": 2},
    "Vowels": {"a": 0, "i": 1, "u": 2}
}