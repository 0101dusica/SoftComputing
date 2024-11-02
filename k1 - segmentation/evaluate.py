# Evaluate the model's accuracy by comparing predicted object counts to ground truth values
def evaluate_model(predictions, ground_truth):
    y_true = [ground_truth[image] for image in predictions]
    y_pred = list(predictions.values())
    absolute_errors = list(map(lambda y: abs(y[0] - y[1]), zip(y_true, y_pred)))
    mean_absolute_error = sum(absolute_errors) / len(absolute_errors)
    return mean_absolute_error