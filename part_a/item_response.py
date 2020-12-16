# import os,sys,inspect
# current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parent_dir = os.path.dirname(current_dir)
# sys.path.insert(0, parent_dir)

from utils import load_train_csv, load_valid_csv, load_public_test_csv, load_train_sparse
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    combined_data = zip(data['is_correct'], data['question_id'], data['user_id'])
    log_lklihood = sum([cij * (theta[i] - beta[j]) - np.log(1 + np.exp(theta[i] - beta[j]))
                        for cij, j, i in combined_data])
    return -log_lklihood


def update_theta_beta(data_by_user, data_by_questions, unique_user_ids, unique_question_ids, lr, theta, beta):
    """ Update theta and beta using gradient descent.
        note that use gradient descent since no close form solution

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    :param data_by_user: {user_id: list, question_id: list,
    is_correct: list} sorted by users
    :param data_by_questions: {user_id: list, question_id: list,
    is_correct: list} sorted by question ids
    :param unique_user_ids: a list of unique user ids in order
    :param unique_question_ids: a list of unique question ids in order
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    # gradient update for each theta
    start_pos = 0
    for i in unique_user_ids:
        theta_i_gradients = []
        for cij, j, i_prime in data_by_user[start_pos:]:
            if i_prime != i:
                break
            jth = -cij + (np.exp(theta[i] - beta[j]) / (1 + np.exp(theta[i] - beta[j])))
            theta_i_gradients.append(jth)
        theta[i] = theta[i] - lr * (sum(theta_i_gradients) / len(theta_i_gradients))
        start_pos += len(theta_i_gradients)
    
    # gradient updates for each beta
    start_pos = 0
    for j in unique_question_ids:
        beta_j_gradients = []
        for cij, j_prime, i in data_by_questions[start_pos:]:
            if j_prime != j:
                break
            ith = cij - (np.exp(theta[i] - beta[j]) / (1 + np.exp(theta[i] - beta[j])))

            beta_j_gradients.append(ith)
        beta[j] = beta[j] - lr * (sum(beta_j_gradients) / len(beta_j_gradients))
        start_pos += len(beta_j_gradients)

    return theta, beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # Initialize theta and beta.
    unique_questions, unique_users = sorted(set(data['question_id'])), sorted(set(data['user_id']))
    data_by_user = sorted(zip(data['is_correct'], data['question_id'], data['user_id']), key=lambda item: (item[-1], item[1]))
    data_by_questions = sorted(zip(data['is_correct'], data['question_id'], data['user_id']), key=lambda item: item[1:])

    np.random.seed(311)
    theta = np.random.uniform(0, 1, size=len(unique_users))
    np.random.seed(412)# np.random.seed(311) 
    beta = np.random.uniform(0, 1, size=len(unique_questions))
    theta, beta = np.array(theta), np.array(beta)
    
    val_acc_lst, per_interation_params = [], []
    train_lld, valid_lld = [], []

    for i in range(iterations): # repeat until convergence
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        print("Iteration - {} NLLK: {} \t Score: {}".format(i, neg_lld, score))

        # for part b plot
        train_lld.append(neg_lld)
        valid_lld.append(neg_log_likelihood(val_data, theta=theta, beta=beta))

        # store the parameters
        val_acc_lst.append(score)
        per_interation_params.append((theta, beta))
        theta, beta = update_theta_beta(data_by_user, data_by_questions, unique_users, unique_questions, lr, theta, beta)

    # get the parameters with the highest validation score
    index = val_acc_lst.index(max(val_acc_lst))
    optimal_theta, optimal_beta = per_interation_params[index]
    return train_lld, valid_lld, optimal_theta, optimal_beta, max(val_acc_lst)


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])

def plot_log_likelihoods(train, validation, train_label, valid_label, figure_name, y_label):
    iterations = [i for i in range(0, len(train))]
    plt.plot(iterations, train, 'b-', label=train_label)
    plt.plot(iterations, validation, 'g-', label=valid_label)
    plt.legend(loc='upper right')
    plt.xlabel('Iterations')
    plt.ylabel(y_label)
    plt.savefig(figure_name)
    plt.close()

def find_probabilities(theta, beta):
    # randomly select 5 question
    np.random.seed(311)
    questions = np.random.randint(0, len(beta), 5)
    sorted_theta = sorted(theta) # sort by the student's ability

    probs = {}
    for j in questions:
        beta_prime = np.repeat(beta[j], len(theta))
        question_correctness = np.exp(sorted_theta - beta_prime) / (1 + np.exp(sorted_theta - beta_prime))
        probs[j] = question_correctness # the prediction trends is based on the sorted student ability
    
    colors = ['b-', 'g-', 'r-', 'm-', 'c-']
    for color, j in enumerate(probs):
        plt.plot(sorted_theta, probs[j], colors[color], label="question {}".format(j))

    plt.legend(loc='upper right')
    plt.xlabel('theta')
    plt.ylabel('Probability of correctness')
    plt.savefig('q2d.png')
    plt.close()

def main():
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    learning_rate, iterations = 0.7, 50

    train_lld, valid_lld, optimal_theta, optimal_beta, validation_acc = irt(train_data, val_data, learning_rate, iterations)

    # part b
    plot_log_likelihoods(train_lld, valid_lld, 
                        'Train', 'Validation',
                        'q2b.png', 'Negative log likelihood')
    # c
    test_acc = evaluate(test_data, optimal_theta, optimal_beta)
    print('Validation accuracy: {}'.format(validation_acc))
    print('Test accuracy: {}'.format(test_acc))

    # d - select the first 5 questions and x axis is students
    find_probabilities(optimal_theta, optimal_beta)

if __name__ == "__main__":
    # os.chdir(os.getcwd() + '/part_a')
    main()
