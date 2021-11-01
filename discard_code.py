
# flaten series into static
# train_x, train_t, train_y = flatten_data(my_dataset, train_indices)  # (1764,713), (1764,), (1764,)
def flatten_data(mdata, data_indices, verbose=1, bool=True):
    x, y = [], []
    uid_list = []
    for idx in data_indices:
        confounder, outcome, uid = mdata[idx][0], mdata[idx][1], mdata[idx][2]
        dx, sex, age = confounder[0], confounder[1], confounder[2]
        # if uid in ['2042577', '1169413']:
        #     print(uid)
        dx = np.sum(dx, axis=0)
        if bool:
            dx = np.where(dx > 0, 1, 0)

        x.append(np.concatenate((dx, [sex], age)))
        y.append(outcome)
        uid_list.append(uid)

    x, y = np.asarray(x), np.asarray(y)
    if verbose:
        d1 = len(dx)
        print('...dx:', x[:, :d1].shape, 'non-zero ratio:', (x[:, :d1] != 0).mean(), 'all-zero:',
              (x[:, :d1].mean(0) == 0).sum())
        print('...all:', x.shape, 'non-zero ratio:', (x != 0).mean(), 'all-zero:', (x.mean(0) == 0).sum())
    return x, y[:, 0], y[:, 1], uid_list
