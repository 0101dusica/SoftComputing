from sklearn.metrics import mean_absolute_error

def evaluate_model(predictions, ground_truth_dict):
    y_true = [ground_truth_dict[image_name] for image_name in predictions.keys()]
    y_pred = list(predictions.values())
    mae = mean_absolute_error(y_true, y_pred)
    return mae