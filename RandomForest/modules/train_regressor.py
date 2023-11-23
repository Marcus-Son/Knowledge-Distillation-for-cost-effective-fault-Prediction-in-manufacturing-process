from sklearn.ensemble import RandomForestRegressor

def train_regressor(train_y):
    return RandomForestRegressor(n_estimators=500, min_samples_split=10, random_state=42)
