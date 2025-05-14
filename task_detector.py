# task_detector.py

def detect_task_type(df, target_column):
    if target_column is None or target_column == "":
        return "UNSUPERVISED"
    
    if df[target_column].dtype == 'object' or len(df[target_column].unique()) <= 10:
        return "CLASSIFICATION"
    else:
        return "REGRESSION"
