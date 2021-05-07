import torch
import numpy as np
from itertools import groupby
import pdb


def get_token_level_ids_probs(ctc_ids, ctc_probs):

    bh = len(ctc_ids[0]) // 2
    y_hat = torch.stack(
        [x[0] for x in groupby(ctc_ids[0])])
    #y_idx = torch.nonzero(y_hat != 0).squeeze(-1)
    #y_hat = y_hat[y_idx]
    probs_hat = []
    cnt = 0
    cidx = 0
    for i, y in enumerate(y_hat.tolist()):
        probs_hat.append(-1)
        while y != ctc_ids[0][cnt]:
            cnt += 1
            if cnt == bh+1:
                cidx = i
        while cnt < ctc_ids.shape[1] and y == ctc_ids[0][cnt]:
            if probs_hat[i] < ctc_probs[0][cnt]:
                probs_hat[i] = ctc_probs[0][cnt].item()
            cnt += 1
            if cnt == bh+1:
                cidx = i
    probs_hat = torch.from_numpy(np.array(probs_hat))

    return y_hat, probs_hat, cidx+1


def tie_breaking(pairs, dtype):
    total_len = len(pairs)
    path = []
    for idx, pair in enumerate(pairs):
        if idx / total_len >= 1/2:
            path.append(pair[1])
        else:
            path.append(pair[0])
    return torch.tensor([path], dtype=dtype)


def dynamic_matching(tensor1, tensor2, prob1=None, prob2=None):
    tensor1 = tensor1.tolist()
    tensor2 = tensor2.tolist()
    M, N = len(tensor1), len(tensor2)
    '''
    if M == 0:
        return None, [(0, t) for t in tensor2], [(0.0, p) for p in prob2], tensor1, tensor2
    if N == 0:
        return None, [(t, 0) for t in tensor1], [(p, 0.0) for p in prob1], tensor1, tensor2
    '''
    dp = [[0 for _ in range(N+1)] for _ in range(M+1)]
    dp[0][0] = 0, [], []
    for i in range(1, N+1):
        dp[0][i] = i, \
            dp[0][i-1][1] + [(0, tensor2[i-1])], \
            dp[0][i-1][2] + [(0, prob2[i-1])]
    for i in range(1, M+1):
        dp[i][0] = i, \
            dp[i-1][0][1] + [(tensor1[i-1], 0)], \
            dp[i-1][0][2] + [(prob1[i-1], 0)]

    for i in range(1, M+1):
        for j in range(1, N+1):
            if tensor1[i-1] == tensor2[j-1]:
                dp[i][j] = dp[i-1][j-1][0], \
                    dp[i-1][j-1][1] + [(tensor1[i-1], tensor2[j-1])], \
                    dp[i-1][j-1][2] + [(prob1[i-1], prob2[j-1])]
            else:
                num, idx = torch.min(torch.tensor([dp[i-1][j-1][0],
                                                   dp[i-1][j][0], dp[i][j-1][0]]), 0)
                dp[i][j] = [0, 0, 0]
                dp[i][j][0] = 1 + num
                if idx == 0:
                    dp[i][j][1] = dp[i-1][j-1][1] + \
                        [(tensor1[i-1], tensor2[j-1])]
                    dp[i][j][2] = dp[i-1][j-1][2] + [(prob1[i-1], prob2[j-1])]
                if idx == 1:
                    dp[i][j][1] = dp[i-1][j][1] + [(tensor1[i-1], 0)]
                    dp[i][j][2] = dp[i-1][j][2] + [(prob1[i-1], 0)]
                if idx == 2:
                    dp[i][j][1] = dp[i][j-1][1] + [(0, tensor2[j-1])]
                    dp[i][j][2] = dp[i][j-1][2] + [(0, prob2[j-1])]
    dp_light = np.array([[n[0] for n in m] for m in dp])
    return dp_light, dp[M][N][1], dp[M][N][2], tensor1, tensor2


def dynamic_matching_xl(t1, t2, prob1, prob2,
                        dp_prev=None, t1_prev=None, t2_prev=None):
    if dp_prev is None:
        return dynamic_matching(t1, t2, prob1, prob2)

    t1 = t1.tolist()
    t2 = t2.tolist()
    prob1 = prob1.tolist()
    prob2 = prob2.tolist()
    '''
    if len(tensor1) == 0:
        return None, [(0, t) for t in tensor2], [(0, p) for p in prob2], tensor1, tensor2
    if len(tensor2) == 0:
        return None, [(t, 0) for t in tensor1], [(p, 0) for p in prob1], tensor1, tensor2
    '''
    sM = len(t1_prev)
    sN = len(t2_prev)
    tensor1 = t1_prev + t1
    tensor2 = t2_prev + t2
    prob1 = [0 for _ in range(sM)] + prob1
    prob2 = [0 for _ in range(sN)] + prob2
    M, N = len(tensor1), len(tensor2)
    dp = [[0 for _ in range(N+1)] for _ in range(M+1)]

    for m in range(sM+1):
        for n in range(sN+1):
            dp[m][n] = dp_prev[m][n], [], []
    for j in range(sN+1, N+1):
        dp[0][j] = dp[0][j-1][0] + 1, [], []
    for i in range(1, sM+1):
        for j in range(sN+1, N+1):
            if tensor1[i-1] == tensor2[j-1]:
                dp[i][j] = dp[i-1][j-1][0], [], []
            else:
                num, idx = torch.min(torch.tensor([dp[i-1][j-1][0],
                                                   dp[i-1][j][0], dp[i][j-1][0]]), 0)
                dp[i][j] = 1 + num.item(), [], []
    for i in range(sM+1, M+1):
        dp[i][0] = dp[i-1][0][0] + 1, [], []
    for i in range(sM+1, M+1):
        for j in range(1, sN+1):
            if tensor1[i-1] == tensor2[j-1]:
                dp[i][j] = dp[i-1][j-1][0], [], []
            else:
                num, idx = torch.min(torch.tensor([dp[i-1][j-1][0],
                                                   dp[i-1][j][0], dp[i][j-1][0]]), 0)
                dp[i][j] = 1 + num.item(), [], []

    for i in range(sM+1, M+1):
        for j in range(sN+1, N+1):
            if tensor1[i-1] == tensor2[j-1]:
                dp[i][j] = dp[i-1][j-1][0],\
                    dp[i-1][j-1][1] + [(tensor1[i-1], tensor2[j-1])],\
                    dp[i-1][j-1][2] + [(prob1[i-1], prob2[j-1])]
            else:
                num, idx = torch.min(torch.tensor([dp[i-1][j-1][0],
                                                   dp[i-1][j][0], dp[i][j-1][0]]), 0)
                dp[i][j] = [0, 0, 0]
                dp[i][j][0] = 1 + num.item()
                if idx == 0:
                    dp[i][j][1] = dp[i-1][j-1][1] +\
                        [(tensor1[i-1], tensor2[j-1])]
                    dp[i][j][2] = dp[i-1][j-1][2] + [(prob1[i-1], prob2[j-1])]
                if idx == 1:
                    dp[i][j][1] = dp[i-1][j][1] + [(tensor1[i-1], 0)]
                    dp[i][j][2] = dp[i-1][j][2] + [(prob1[i-1], 0)]
                if idx == 2:
                    dp[i][j][1] = dp[i][j-1][1] + [(0, tensor2[j-1])]
                    dp[i][j][2] = dp[i][j-1][2] + [(0, prob2[j-1])]
    dp_light = np.array([[n[0] for n in m] for m in dp])

    reserve_dim = 5
    dim1, dim2 = max(0, M - reserve_dim), max(0, N - reserve_dim)
    dp_light = dp_light[dim1:, dim2:]
    tensor1 = tensor1[dim1:]
    tensor2 = tensor2[dim2:]
    pdb.set_trace()
    return dp_light, dp[M][N][1], dp[M][N][2], tensor1, tensor2
