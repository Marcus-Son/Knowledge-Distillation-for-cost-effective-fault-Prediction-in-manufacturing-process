
def test_predict_kd(random_forest_r,test_X_new):
    y_predictions = random_forest_r.predict(test_X_new)

    return y_predictions