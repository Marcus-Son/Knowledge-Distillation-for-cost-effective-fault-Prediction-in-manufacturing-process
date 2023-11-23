def test_predict_vm(random_forest_ad,test_X_new):
    y_predictions = random_forest_ad.predict_proba(test_X_new)[:,1]
    return y_predictions