def test_predict(random_forest_basic,test_X_new):
    y_predictions = random_forest_basic.predict_proba(test_X_new)[:,1]

    return y_predictions