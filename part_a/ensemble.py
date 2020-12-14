import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from utils import load_train_csv, load_valid_csv, load_public_test_csv
import numpy as np
import matplotlib.pyplot as plt
from item_response import \
    irt as base_classifer, \
    evaluate as base_evaluate, \
    sigmoid # this is the base classifer

def bootstrapping(data, num_learners=3):
    """bootstrapping from the training dataset with replacement
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param num_learners: number of base learner
    :return a list of bootstrapped dataset (in dictionary form)
    """
    data['user_id'] = np.array(data['user_id'])
    data['question_id'] = np.array(data['question_id'])
    data['is_correct'] = np.array(data['is_correct'])

    # bootstrap
    bootstrapped_data = []
    for _ in range(num_learners):
        bootstrap = {}
        # sample data with replacement
        bootstrapped_indices = np.random.randint(low=0, high=len(data['user_id']), size=len(data['user_id']))
        # check if sample with replacement =>  len(set(bootstrapped_indices)) < len(data['user_id'])
        bootstrap['user_id'] = data['user_id'][bootstrapped_indices]
        bootstrap['question_id'] = data['question_id'][bootstrapped_indices]
        bootstrap['is_correct'] = data['is_correct'][bootstrapped_indices]
        bootstrapped_data.append(bootstrap)
    return bootstrapped_data

def train_base_classifiers(bootstrap_data, val_data, lr=8, iteration=10): # lr=7, iteration=50
    """train each base classifer

    :param bootstrap_data: a list of bagged data set
    :param val_data: validation data set
    :param lr: the learning rate
    :param iteration: the number of iteration to train each base model
    :return a dictionary of trained base model
    """
    # train the base classifier
    trained_models = {}
    for model, bootstrap in enumerate(bootstrap_data):
        _, _, opt_theta, opt_beta, validation_acc = base_classifer(bootstrap, val_data, lr, iteration)
        trained_models[model] = [opt_theta, opt_beta, validation_acc]
    return trained_models

def bag_evaluate(data, trained_model):
    predictions = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        sum_prediction = 0

        for model in trained_model:
            theta, beta= trained_model[model][0], trained_model[model][1]
            x = (theta[u] - beta[q]).sum()
            p_a = sigmoid(x) # the prediction for this base classifier
            correct = p_a >= 0.5
            sum_prediction += correct

        # get the bagged prediction
        bagged_pred = sum_prediction / len(trained_model)
        predictions.append(bagged_pred >= 0.5)
    return np.sum((data["is_correct"] == np.array(predictions))) / len(data["is_correct"])

def main():
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    bootstrapped_data = bootstrapping(train_data)
    trained_models = train_base_classifiers(bootstrapped_data, val_data)

    # bagged ensemble accuracy on validation and test set
    valid_acc = bag_evaluate(val_data, trained_models)
    test_acc = bag_evaluate(test_data, trained_models)
    print('Ensemble validation accuracy: {}'.format(valid_acc))
    print('Ensemble test accuracy: {}'.format(test_acc))

    # base model accuracy on validation and test set
    # for i in trained_models:
    #     theta, beta = trained_models[i][0], trained_models[i][1]
    #     print('Base classifier {} validation accuracy: {}'.format(i, base_evaluate(val_data, theta, beta)))
    #     print('Base classifier {} test accuracy: {}'.format(i, base_evaluate(test_data, theta, beta)))

if __name__ == "__main__":
    os.chdir(os.getcwd() + '/part_a')
    main()
    # on average, ensemble is slightly better
        # could be even better, but constrained by the correlation between dataset
        # from the equation of the bagged variance, we know that it is more stable as long as there are some randomnesss
    