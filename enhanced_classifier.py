# import os,sys,inspect
# current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parent_dir = os.path.dirname(current_dir)
# sys.path.insert(0, parent_dir)

from utils import *
import numpy as np
import matplotlib.pyplot as plt
from part_a.item_response import irt, sigmoid
from part_a.neural_network import load_data, evaluate
from part_a.neural_network import AutoEncoder
from collections import Counter

import torch
import torch.optim as optim
from torch.autograd import Variable
import copy

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
        bootstrapped_indices = np.random.randint(low=0, high=len(data['user_id']) - 1, size=len(data['user_id']))
        # check if sample with replacement =>  len(set(bootstrapped_indices)) < len(data['user_id'])
        bootstrap['user_id'] = data['user_id'][bootstrapped_indices]
        bootstrap['question_id'] = data['question_id'][bootstrapped_indices]
        bootstrap['is_correct'] = data['is_correct'][bootstrapped_indices]
        bootstrapped_data.append(bootstrap)
    return bootstrapped_data

def train(model, lr, lamb, train_data, zero_train_data, valid_data, bootstrap_entries, num_epoch):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: None
    """

    train_loss_lst = []
    valid_acc_lst = []
    best_acc = 0
    best_model = None
    best_epoch = 0
    # Tell PyTorch you are training the model.
    model.train()

    data_count = dict(Counter(bootstrap_entries))
    unique_entries = sorted(list(data_count.keys())) # sort by user than question id

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]

    for epoch in range(0, num_epoch):
        train_loss = 0.
        start = 0
        for user_id in range(num_student):
            # replace this with the sampled one and other entries = 0
            questions, counts = [], []
            for i, uq in enumerate(unique_entries[start:]):
                if uq[0] != user_id:
                    break
                counts.append(data_count[(user_id, uq[1])]) # get the count of that question
                questions.append(uq[1])
            start += len(questions)
            sampled_inputs = np.empty(train_data.shape[1])
            sampled_inputs[:] = np.NaN

            # get the sampled entries
            sampled_inputs[questions] = zero_train_data[user_id, questions]
            zero_inputs = np.copy(sampled_inputs)
            zero_inputs[np.isnan(sampled_inputs)] = 0 # replace the nan with 0

            inputs = Variable(torch.FloatTensor(zero_inputs)).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(torch.FloatTensor(sampled_inputs).unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]

            # modify it so that it consider the repeated sample
            # the repeated entries and target need to multiple the mulplicity to take the replacement into account
            output_mulplicity = np.empty(output.shape[1])
            output_mulplicity[:] = 1
            output_mulplicity[questions] = counts # the multiplicity of each question after resample
            output, target = output * torch.FloatTensor(output_mulplicity).unsqueeze(0), target * torch.FloatTensor(output_mulplicity).unsqueeze(0)
            loss = torch.sum((output - target) ** 2.) + 0.5 * lamb * model.get_weight_norm()
            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        valid_acc = evaluate(model, zero_train_data, valid_data)
        print("Epoch: {} \tTraining Cost: {:.6f}\t "
              "Valid Acc: {}".format(epoch, train_loss, valid_acc))

        train_loss_lst.append(train_loss)
        valid_acc_lst.append(valid_acc)
        if valid_acc > best_acc:
            best_acc = valid_acc
            best_model = copy.deepcopy(model)
            best_epoch = epoch

    print("Best epoch = {}".format(best_epoch))
    return max(valid_acc_lst), best_model

def nn_classifier(zero_train_matrix, train_matrix, valid_data, test_data, bootstrap, k=10, lamb=0.01, num_epoch=100, lr=0.01):
    """insert the optimal hyperparameters"""

    model = AutoEncoder(train_matrix.shape[1], k=k)
    max_valid_acc, best_model = train(model, lr, lamb, train_matrix, zero_train_matrix, valid_data, bootstrap, num_epoch)
    # test_acc = evaluate(best_model, zero_train_matrix, test_data)
    return max_valid_acc, best_model

def mix_ensemble():
    'mix of weighted irt and nn models'
    zero_train_matrix, train_matrix, valid_data, test_data, train_data = load_data('./data')
    global irt

    irt_bootstraps = bootstrapping(train_data, 7) # 7
    trained_irts = {}
    irt_acc = 0
    for model, bootstrap in enumerate(irt_bootstraps):
        _, _, opt_theta, opt_beta, validation_acc = irt(bootstrap, valid_data, 8, 5) #8， 10
        trained_irts[model] = [opt_theta, opt_beta, validation_acc]
        irt_acc += validation_acc
    irt_avg_acc = irt_acc / len(irt_bootstraps)

    # train nn models
    nn_bootstraps = bootstrapping(train_data, 3)
    trained_nn = {}
    nn_acc = 0
    for model, bootstrap in enumerate(nn_bootstraps):
        zipped_data = [(u, q) for q, u in zip(bootstrap['question_id'], bootstrap['user_id'])]
        from os import path
        model_path = './nn_models/model{}'.format(model)
        if path.exists(model_path): # load model from file
            opt_model = torch.load(model_path)
            opt_model.eval()
            opt_acc = evaluate(opt_model, zero_train_matrix, valid_data)
        else: # train the model
            opt_acc, opt_model = nn_classifier(zero_train_matrix, train_matrix, valid_data, test_data, zipped_data)
            torch.save(opt_model, './nn_models/model{}'.format(model))
        trained_nn[model] = opt_model
        nn_acc += opt_acc
    nn_avg_acc = nn_acc / len(nn_bootstraps)

    weighted_predictions, unweighted_predictions = mix_ensemble_predictions(test_data, trained_irts, trained_nn, zero_train_matrix, irt_avg_acc, nn_avg_acc)
    # find accuracy for the public dataset
    final_weighted_acc = np.sum((test_data["is_correct"] == np.array(weighted_predictions))) / len(test_data["is_correct"])
    final_unweigthed_acc = np.sum((test_data["is_correct"] == np.array(unweighted_predictions))) / len(test_data["is_correct"])
    print("Weighted Mix Ensemble Test Accuracy: {}".format(final_weighted_acc))
    print("Unweighted Mix Ensemble Test Accuracy: {}".format(final_unweigthed_acc))

    # Load the private test dataset.
    private_test = load_private_test_csv("./data")
    pri_weighted_predicts, _ = mix_ensemble_predictions(private_test, trained_irts, trained_nn, zero_train_matrix, irt_avg_acc, nn_avg_acc)
    private_test['is_correct'] = pri_weighted_predicts
    save_private_test_csv(private_test)


def mix_ensemble_predictions(test_data, trained_irts, trained_nn, zero_train_matrix, irt_avg_acc, nn_avg_acc):
    # prediction of each question for irt model
    irt_predictions = []
    for i, q in enumerate(test_data['question_id']):
        u = test_data["user_id"][i]
        bag_predictions = []
        for irt in trained_irts:
            theta, beta = trained_irts[irt][0], trained_irts[irt][1]
            x = (theta[u] - beta[q]).sum()
            prediction = sigmoid(x) >= 0.5  #the prediction for this base classifier
            bag_predictions.append(prediction)
        irt_predictions.append(bag_predictions)

    # prediction of each question for nn model
    nn_predictions = []
    for i, u in enumerate(test_data["user_id"]):
        inputs = Variable(zero_train_matrix[u]).unsqueeze(0)
        bag_predictions = []
        for model in trained_nn:
            nn = trained_nn[model]
            nn.eval()
            output = nn(inputs)
            # get the corresponding question prediction
            prediction = output[0][test_data["question_id"][i]].item() >= 0.5
            bag_predictions.append(prediction)
        nn_predictions.append(bag_predictions)

    # mix evaluate
    weighted_predictions, unweighted_predictions = [], []
    acc = np.array([irt_avg_acc, nn_avg_acc])
    weights = acc / np.sum(acc) # normlized weights
    for irt_predicts, nn_predicts in zip(irt_predictions, nn_predictions):
        irt, nn = sum(irt_predicts) / len(irt_predicts), sum(nn_predicts) / len(nn_predicts)
        prediction = np.array([irt, nn])
        weighted_predictions.append(np.dot(weights, prediction) >= 0.5) # weight the prediction before evaluate
        unweighted_predictions.append(np.average(prediction) >= 0.5)

    return weighted_predictions, unweighted_predictions


def nn_ensemble():
    zero_train_matrix, train_matrix, valid_data, test_data, train_data = load_data('./data')
    bootstraps = bootstrapping(train_data, 3)
    # ensemble the nn
    models = []
    for i, data in enumerate(bootstraps):
        zipped_data = [(u, q) for q, u in zip(data['question_id'], data['user_id'])]
        opt_acc, opt_model = nn_classifier(zero_train_matrix, train_matrix, valid_data, test_data, zipped_data)
        # save the trained model for future use
        torch.save(opt_model, './nn_models/model{}'.format(i))
        models.append(opt_model)
    # check test accuracy
    bag_evaluate(models, zero_train_matrix, test_data)
    return models


def bag_evaluate(trained_model, train_data, valid_data):
    # Tell PyTorch you are evaluating the model.
    total = 0
    correct = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        member_correct = 0
        for model in trained_model:
            model.eval()
            output = model(inputs)

            guess = output[0][valid_data["question_id"][i]].item() >= 0.5
            if guess == valid_data["is_correct"][i]:
                member_correct += 1

        bagged_prediction = member_correct / len(trained_model)
        guess = bagged_prediction >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    acc = correct / float(total)
    print("bag test acc: {}".format(acc))
    return acc

if __name__ == "__main__":
    # os.chdir(os.getcwd() + '/part_a')
    # nn_ensemble()
    mix_ensemble()
