import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from utils import *
import numpy as np
import matplotlib.pyplot as plt
from part_a import item_response as irt
# from part_a.ensemble import bootstrapping
from part_a.neural_network import load_data, evaluate
from part_a.deep_nn import AutoEncoder
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
        bootstrapped_indices = np.random.randint(low=0, high=len(data['user_id']), size=len(data['user_id']))
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
            #TODO: replace this with the sampled one and other entries = 0
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
            # nan_mask = np.isnan(sampled_inputs.reshape(-1,1))
            nan_mask = np.isnan(torch.FloatTensor(sampled_inputs).unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]

            #TODO: modify it so that it consider the repeated sample
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

def nn_ensemble():
    zero_train_matrix, train_matrix, valid_data, test_data, train_data = load_data('./data')
    bootstraps = bootstrapping(train_data, 7)
    # ensemble the nn
    models = []
    for i, data in enumerate(bootstraps):
        zipped_data = [(u, q) for q, u in zip(data['question_id'], data['user_id'])]
        # Model class must be defined somewhere
        # model = torch.load(PATH)
        # model.eval()
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

def mix_ensemble():
    return

if __name__ == "__main__":
    # os.chdir(os.getcwd() + '/part_a')
    # main()
    nn_ensemble()
