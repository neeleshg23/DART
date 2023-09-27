import numpy as np
# ================================================================ metrics

def _cossim(Y, Y_hat):
    ynorm = np.linalg.norm(Y) + 1e-20
    yhat_norm = np.linalg.norm(Y_hat) + 1e-20
    return ((Y / ynorm) * (Y_hat / yhat_norm)).sum()

def _compute_metrics(task, Y_hat):
    Y = task.Y_test
    diffs = Y - Y_hat
    raw_mse = np.mean(diffs * diffs)
    normalized_mse = raw_mse / np.var(Y)
    # Y_meannorm = Y - Y.mean()
    # Y_hat_meannorm = Y_hat - Y_hat.mean()
    # ynorm = np.linalg.norm(Y_meannorm) + 1e-20
    # yhat_norm = np.linalg.norm(Y_hat_meannorm) + 1e-20
    # r = ((Y_meannorm / ynorm) * (Y_hat_meannorm / yhat_norm)).sum()
    metrics = {'raw_mse': raw_mse, 'normalized_mse': normalized_mse,
               'corr': _cossim(Y - Y.mean(), Y_hat - Y_hat.mean()),
               'cossim': _cossim(Y, Y_hat),  # 'bias': diffs.mean(),
               'y_mean': Y.mean(), 'y_std': Y.std(),
               'yhat_std': Y_hat.std(), 'yhat_mean': Y_hat.mean()}

    problem = task.info['problem']
    metrics['problem'] = problem
    if problem == 'softmax':
        lbls = task.info['lbls_test'].astype(np.int32)
        b = task.info['biases']
        logits_amm = Y_hat + b
        logits_orig = Y + b
        lbls_amm = np.argmax(logits_amm, axis=1).astype(np.int32)
        lbls_orig = np.argmax(logits_orig, axis=1).astype(np.int32)
        metrics['acc_amm'] = np.mean(lbls_amm == lbls)
        metrics['acc_orig'] = np.mean(lbls_orig == lbls)

    return metrics
