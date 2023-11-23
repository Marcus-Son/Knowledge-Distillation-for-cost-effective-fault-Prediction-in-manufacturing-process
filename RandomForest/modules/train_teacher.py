from sklearn.ensemble import RandomForestClassifier

def train_teacher(train_y):
    return RandomForestClassifier(n_estimators=500, min_samples_split=10, random_state=42)
