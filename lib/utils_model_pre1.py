import pickle

import numpy as np
import csv
import numpy as np
from scipy.sparse.linalg import eigs
import torch
# from metrics import mean_absolute_error, mean_squared_error, masked_mape_np
# from lib.metrics import *
# from metrics import *
# from baseMay_pre6.lib.metrics import mean_absolute_percentage_error, get_MAE, get_RMSE



#正确的指标计算方式
def mean_absolute_percentage_error(y_pred,y_true):
    idx=np.nonzero(y_true)
    return np.mean(np.abs((y_true[idx]-y_pred[idx])/y_true[idx]))
def get_MSE(pred,real):
    return np.mean(np.power(real-pred,2))
def get_MAE(pred,real):
    # print('真实值')
    # print(pred.shape)#(388, 80, 80)
    # print(real.shape)#(388, 80, 80)
    # print(real[33][1])
    # print(real[111][1])
    return np.mean(np.abs(real-pred))

def get_RMSE(pred,real):
    return np.sqrt(get_MSE(pred=pred,real=real))

class StandardScaler_Torch:
    """
    Standard the input
    """

    def __init__(self, mean, std, device):
        self.mean = torch.tensor(data=mean, dtype=torch.float, device=device)
        self.std = torch.tensor(data=std, dtype=torch.float, device=device)
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

def predict(net, test_loader, device):
    '''
    predict

    Parameters
    ----------
    net: model

    test_loader: gluon.data.DataLoader

    Returns
    ----------
    prediction: np.ndarray,
                shape is (num_of_samples, num_of_vertices, num_for_predict)

    '''
    net.eval()
    with torch.no_grad():
        prediction = []
        for index, (test_w, test_d, test_r, test_t,test_w_toweek,test_w_tohour,
                    test_d_toweek,test_d_tohour,test_r_toweek,test_r_tohour) in enumerate(test_loader):
            test_w = test_w.to(device)
            test_d = test_d.to(device)
            test_r = test_r.to(device)
            test_t = test_t.to(device)
            test_w_toweek = test_w_toweek.to(device)
            test_w_tohour = test_w_tohour.to(device)
            test_d_toweek = test_d_toweek.to(device)
            test_d_tohour = test_d_tohour.to(device)
            test_r_toweek = test_r_toweek.to(device)
            test_r_tohour = test_r_tohour.to(device)


            output= net([test_w, test_d, test_r],[test_w_toweek,test_w_tohour,
                    test_d_toweek,test_d_tohour,test_r_toweek,test_r_tohour])
            prediction.append(output.cpu().detach().numpy())

        prediction = np.concatenate(prediction, 0)
        return prediction


def evaluate(net, test_loader, true_value, device, epoch,scaler):
    '''
    compute MAE, RMSE, MAPE scores of the prediction
    for 3, 6, 12 points on testing set

    Parameters
    ----------
    net: model

    test_loader: gluon.data.DataLoader

    true_value: np.ndarray, all ground truth of testing set
                shape is (num_of_samples, num_for_predict, num_of_vertices)

    num_of_vertices: int, number of vertices

    epoch: int, current epoch

    '''
    scaler = scaler
    scaler_torch = StandardScaler_Torch(scaler.mean, scaler.std, device=device)
    net.eval()
    with torch.no_grad():
        prediction= predict(net, test_loader,device)
        prediction = scaler_torch.inverse_transform(prediction)  # 是将标准化后的数据转换为原始数据  #(316, 80, 80, 6)
        true_value = scaler_torch.inverse_transform(true_value)  # (316, 80, 80, 6)
        # print('prediction')
        # print(prediction.shape)
        # print(true_value.shape)

        mae = get_MAE(prediction, true_value)
        rmse = get_RMSE(prediction, true_value)
        print('test Average Horizon, MAE: %.6f, RMSE: %.6f' % (
            mae, rmse))
        return mae,rmse



def scaled_Laplacian(W):
    '''
    compute \tilde{L}

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    scaled_Laplacian: np.ndarray, shape (N, N)

    '''

    assert W.shape[0] == W.shape[1]

    D = np.diag(np.sum(W, axis=1))

    L = D - W

    lambda_max = eigs(L, k=1, which='LR')[0].real

    return (2 * L) / lambda_max - np.identity(W.shape[0])


def cheb_polynomial(L_tilde, K):
    '''
    compute a list of chebyshev polynomials from T_0 to T_{K-1}

    Parameters
    ----------
    L_tilde: scaled Laplacian, np.ndarray, shape (N, N)

    K: the maximum order of chebyshev polynomials

    Returns
    ----------
    cheb_polynomials: list[np.ndarray], length: K, from T_0 to T_{K-1}

    '''

    N = L_tilde.shape[0]

    cheb_polynomials = [np.identity(N), L_tilde.copy()]

    for i in range(2, K):
        cheb_polynomials.append(
            2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])

    return cheb_polynomials


def compute_val_loss(net, val_loader,true_val_value, loss_function,device, epoch,scaler):
    '''
    compute mean loss on validation set

    Parameters
    ----------
    net: model

    val_loader: gluon.data.DataLoader

    loss_function: func

    epoch: int, current epoch

    '''
    scaler = scaler
    scaler_torch = StandardScaler_Torch(scaler.mean, scaler.std, device=device)
    net.eval()
    with torch.no_grad():
        prediction= predict(net, val_loader, device)
        prediction = scaler_torch.inverse_transform(prediction)  # 是将标准化后的数据转换为原始数据  #(316, 80, 80, 6)
        true_value = scaler_torch.inverse_transform(true_val_value)  # (316, 80, 80, 6)
        # print('############')
        # print(prediction.shape)
        # print(true_value.shape)

        mae = get_MAE(prediction, true_value)
        rmse = get_RMSE(prediction, true_value)
        print('val Average Horizon, MAE: %.6f, RMSE: %.6f' % (
            mae, rmse))

        tmp = []
        for index, (val_w, val_d, val_r, val_t,val_w_toweek,val_w_tohour,
                    val_d_toweek,val_d_tohour,val_r_toweek,val_r_tohour) in enumerate(val_loader):
            val_w = val_w.to(device)
            val_d = val_d.to(device)
            val_r = val_r.to(device)
            val_t = val_t.to(device)
            val_w_toweek = val_w_toweek.to(device)
            val_w_tohour = val_w_tohour.to(device)
            val_d_toweek = val_d_toweek.to(device)
            val_d_tohour = val_d_tohour.to(device)
            val_r_toweek = val_r_toweek.to(device)
            val_r_tohour = val_r_tohour.to(device)
            output= net([val_w, val_d, val_r],[val_w_toweek,val_w_tohour,
                    val_d_toweek,val_d_tohour,val_r_toweek,val_r_tohour])
            output = scaler_torch.inverse_transform(output)  # 是将标准化后的数据转换为原始数据
            val_t = scaler_torch.inverse_transform(val_t)
            # print('output')
            # print(output.shape)
            l = loss_function(output, val_t)
            tmp.append(l.item())

        validation_loss = sum(tmp) / len(tmp)

        print('epoch: %s, validation loss: %.2f' % (epoch, validation_loss))
        return validation_loss,mae, rmse

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data
def load_graph_data_hz(pkl_filename):
    adj_mx = load_pickle(pkl_filename)  #hz.shape:(80,80)
    return adj_mx.astype(np.float32)



def normalization(train, val, test):
    '''
    Parameters
    ----------
    train, val, test: np.ndarray

    Returns
    ----------
    stats: dict, two keys: mean and std

    train_norm, val_norm, test_norm: np.ndarray,
                                     shape is the same as original

    '''

    assert train.shape[1:] == val.shape[1:] and val.shape[1:] == test.shape[1:]

    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)

    def normalize(x):
        return (x - mean) / std

    train = (train).transpose(0,2,1,3)
    val = (val).transpose(0,2,1,3)
    test =(test).transpose(0,2,1,3)

    return {'mean': mean, 'std': std}, train, val, test

def search_data(sequence_length, num_of_batches, label_start_idx,
                num_for_predict, units, points_per_hour):

    if points_per_hour < 0:
        raise ValueError("points_per_hour should be greater than 0!")

    if label_start_idx + num_for_predict > sequence_length:
        return None

    x_idx = []
    for i in range(1, num_of_batches + 1):
        start_idx = label_start_idx - points_per_hour * units * i
        end_idx = start_idx + num_for_predict
        if start_idx >= 0:
            x_idx.append((start_idx, end_idx))
        else:
            return None

    if len(x_idx) != num_of_batches:
        return None

    return x_idx[::-1]#倒叙输出,符合时间的 顺序输出,这里不占用多少空间

def search_datah(sequence_length, num_of_batches, label_start_idx,
                num_for_predict, units, points_per_hour):

    if points_per_hour < 0:
        raise ValueError("points_per_hour should be greater than 0!")

    if label_start_idx + num_for_predict > sequence_length:
        return None

    x_idx = []
    for i in range(1, num_of_batches + 1):
        start_idx = label_start_idx - num_for_predict * units * i
        end_idx = start_idx + num_for_predict
        if start_idx >= 0:
            x_idx.append((start_idx, end_idx))
        else:
            return None

    if len(x_idx) != num_of_batches:
        return None

    return x_idx[::-1]#倒叙输出,符合时间的 顺序输出,这里不占用多少空间

def get_sample_indices(data_sequence,Metro_week_matrix,Metro_hour_matrix, num_of_weeks, num_of_days, num_of_hours,
                       label_start_idx, num_for_predict, points_per_hour=12):

    week_indices = search_data(data_sequence.shape[0], num_of_weeks,
                               label_start_idx, num_for_predict,
                               7 * 24, points_per_hour)
    if not week_indices:
        return None

    day_indices = search_data(data_sequence.shape[0], num_of_days,
                              label_start_idx, num_for_predict,
                              24, points_per_hour)
    if not day_indices:
        return None

    hour_indices = search_datah(data_sequence.shape[0], num_of_hours,
                               label_start_idx, num_for_predict,
                               1, points_per_hour)
    if not hour_indices:
        return None

        ####-----------对week---------------
    week_sample = np.concatenate([data_sequence[i:j] for i, j in week_indices], axis=0)  # week_indices:[(0, 6)]
    week_sample_toweek = np.concatenate([Metro_week_matrix[i:j] for i, j in week_indices], axis=0)
    week_sample_tohour = np.concatenate([Metro_hour_matrix[i:j] for i, j in week_indices], axis=0)
    ##------------对day--------------------
    day_sample = np.concatenate([data_sequence[i:j] for i, j in day_indices],
                                axis=0)  # day_indices:[(864, 870), (720, 726), (576, 582)]
    day_sample_toweek = np.concatenate([Metro_week_matrix[i:j] for i, j in day_indices],
                                       axis=0)
    day_sample_tohour = np.concatenate([Metro_hour_matrix[i:j] for i, j in day_indices],
                                       axis=0)
    ###-------------对hour-----------------
    hour_sample = np.concatenate([data_sequence[i:j] for i, j in hour_indices],
                                 axis=0)  # hour_indices:[(1002, 1008), (996, 1002), (990, 996)]
    hour_sample_toweek = np.concatenate([Metro_week_matrix[i:j] for i, j in hour_indices],
                                        axis=0)
    hour_sample_tohour = np.concatenate([Metro_hour_matrix[i:j] for i, j in hour_indices],
                                        axis=0)
    target = data_sequence[label_start_idx:label_start_idx + num_for_predict]

    return week_sample, day_sample, hour_sample, target, week_sample_toweek, week_sample_tohour, day_sample_toweek, day_sample_tohour, hour_sample_toweek, hour_sample_tohour


def read_and_generate_dataset(Metro_edge_matrix,Metro_week_matrix,Metro_hour_matrix,
                              num_of_weeks, num_of_days,
                              num_of_hours, num_for_predict,
                              points_per_hour=6, merge=False,scaler_axis=(0,
                                                     1,
                                                     2,
                                                     3)):



    all_samples = []
    for idx in range(Metro_edge_matrix.shape[0]):
        sample = get_sample_indices(Metro_edge_matrix, Metro_week_matrix,Metro_hour_matrix,num_of_weeks, num_of_days,
                                    num_of_hours, idx, num_for_predict,
                                    points_per_hour)
        if not sample:
            continue

        week_sample, day_sample, hour_sample, target, week_sample_toweek, week_sample_tohour, \
        day_sample_toweek, day_sample_tohour, hour_sample_toweek, hour_sample_tohour = sample

        all_samples.append((
            np.expand_dims(week_sample, axis=0).transpose((0, 2, 3, 1)),
            np.expand_dims(day_sample, axis=0).transpose((0, 2, 3, 1)),
            np.expand_dims(hour_sample, axis=0).transpose((0, 2, 3, 1)),
            np.expand_dims(target, axis=0).transpose((0, 2, 3, 1)),
            np.expand_dims(week_sample_toweek, axis=0),  # (6,7)
            np.expand_dims(week_sample_tohour, axis=0),  # (6,144)
            np.expand_dims(day_sample_toweek, axis=0),
            np.expand_dims(day_sample_tohour, axis=0),
            np.expand_dims(hour_sample_toweek, axis=0),
            np.expand_dims(hour_sample_tohour, axis=0)
        ))

    split_line1 = int(len(all_samples) * 0.6)
    split_line2 = int(len(all_samples) * 0.8)

    if not merge:
        training_set = [np.concatenate(i, axis=0)
                        for i in zip(*all_samples[:split_line1])]
    else:
        print('Merge training set and validation set!')
        training_set = [np.concatenate(i, axis=0)
                        for i in zip(*all_samples[:split_line2])]

    validation_set = [np.concatenate(i, axis=0)
                      for i in zip(*all_samples[split_line1: split_line2])]
    testing_set = [np.concatenate(i, axis=0)
                   for i in zip(*all_samples[split_line2:])]

    train_week, train_day, train_hour, train_target, train_week_toweek, train_week_tohour, \
    train_day_toweek, train_day_tohour, train_hour_toweek, train_hour_tohour, = training_set
    val_week, val_day, val_hour, val_target, val_week_toweek, val_week_tohour, \
    val_day_toweek, val_day_tohour, val_hour_toweek, val_hour_tohour = validation_set
    test_week, test_day, test_hour, test_target, test_week_toweek, test_week_tohour, \
    test_day_toweek, test_day_tohour, test_hour_toweek, test_hour_tohour = testing_set

    print('training data: week: {}, day: {}, recent: {}, target: {}'.format(
        train_week.shape, train_day.shape,
        train_hour.shape, train_target.shape))
    print('validation data: week: {}, day: {}, recent: {}, target: {}'.format(
        val_week.shape, val_day.shape, val_hour.shape, val_target.shape))
    print('testing data: week: {}, day: {}, recent: {}, target: {}'.format(
        test_week.shape, test_day.shape, test_hour.shape, test_target.shape))

    print('外部因素')
    print(
        'training external: week_toweek:{},week_tohour:{},day_toweek:{},day_tohour:{},recent_toweek:{},recent_tohour:{}'.format(
            train_week_toweek.shape, train_week_tohour.shape, train_day_toweek.shape, train_day_tohour.shape,
            train_hour_toweek.shape, train_hour_tohour.shape
        ))
    print(
        'validation external: week_toweek:{},week_tohour:{},day_toweek:{},day_tohour:{},recent_toweek:{},recent_tohour:{}'.format(
            val_week_toweek.shape, val_week_tohour.shape, val_day_toweek.shape, val_day_tohour.shape,
            val_hour_toweek.shape, val_hour_tohour.shape
        ))
    print(
        'testing external: week_toweek:{},week_tohour:{},day_toweek:{},day_tohour:{},recent_toweek:{},recent_tohour:{}'.format(
            test_week_toweek.shape, test_week_tohour.shape, test_day_toweek.shape, test_day_tohour.shape,
            test_hour_toweek.shape, test_hour_tohour.shape
        ))

    # 标准化
    scaler = StandardScaler(mean=train_day.mean(axis=scaler_axis),  # 标准化
                            std=train_day.std(axis=scaler_axis))


    train_week_norm, val_week_norm, test_week_norm = scaler.transform(train_week), scaler.transform(
        val_week), scaler.transform(test_week)
    train_day_norm, val_day_norm, test_day_norm = scaler.transform(train_day), scaler.transform(
        val_day), scaler.transform(test_day)
    train_recent_norm, val_recent_norm, test_recent_norm = scaler.transform(train_hour), scaler.transform(
        val_hour), scaler.transform(test_hour)
    train_target, val_target, test_target = scaler.transform(train_target), scaler.transform(
        val_target), scaler.transform(test_target)

    #----------------------------################
    # train_week_norm, val_week_norm, test_week_norm = train_week, val_week, test_week
    # train_day_norm, val_day_norm, test_day_norm = train_day,val_day,test_day
    # train_recent_norm, val_recent_norm, test_recent_norm = train_hour,val_hour,test_hour
    # train_target, val_target, test_target = train_target, val_target,test_target
    # print('##########3333#########')
    # print(val_target.shape)
    # # print(prediction.shape)
    # print('------real,timestep=42,station15-station9-----')
    # print(val_target[42][15][9])
    # print('------prediction,timestep=42,station15-station9-----')
    # print(test_target[42][15][9])
    # print('------real,timestep=108,station15-station9-----')
    # print(val_recent_norm[108][15][9])
    # print('------prediction,timestep=108,station15-station9-----')

    all_data = {
        'train': {
            'week': train_week_norm,
            'day': train_day_norm,
            'recent': train_recent_norm,
            'target': train_target,
            'week_toweek': train_week_toweek,
            'week_tohour': train_week_tohour,
            'day_toweek': train_day_toweek,
            'day_tohour': train_day_tohour,
            'recent_toweek': train_hour_toweek,
            'recent_tohour': train_hour_tohour
        },
        'val': {
            'week': val_week_norm,
            'day': val_day_norm,
            'recent': val_recent_norm,
            'target': val_target,
            'week_toweek': val_week_toweek,
            'week_tohour': val_week_tohour,
            'day_toweek': val_day_toweek,
            'day_tohour': val_day_tohour,
            'recent_toweek': val_hour_toweek,
            'recent_tohour': val_hour_tohour
        },
        'test': {
            'week': test_week_norm,
            'day': test_day_norm,
            'recent': test_recent_norm,
            'target': test_target,
            'week_toweek': test_week_toweek,
            'week_tohour': test_week_tohour,
            'day_toweek': test_day_toweek,
            'day_tohour': test_day_tohour,
            'recent_toweek': test_hour_toweek,
            'recent_tohour': test_hour_tohour
        },
        # 'stats': {
        #     'week': week_stats,
        #     'day': day_stats,
        #     'recent': recent_stats
        # }
    }

    return all_data,scaler