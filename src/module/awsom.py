import copy

import numpy as np
import pywt


class AWSOM:
    def __init__(self, n_ls=(6, 4, 2), dwt_type="haar") -> None:
        """
        discover interesting patters and trends

        Attributes
        -------------------
        """
        if dwt_type != "haar":
            raise NotImplementedError
        self.n_ls = n_ls
        self.dwt_type = dwt_type
        self.depth = len(n_ls)
        self.k = sum(n_ls)

    def init_params(self, sig):
        self.current_t = len(sig)
        self.max_depth = int(np.log2(len(sig)))
        self.level = self.max_depth - self.depth + 1
        self.P = [np.zeros((self.k, self.k)) for _ in range(self.level)]
        self.q = [np.zeros((self.k,)) for _ in range(self.level)]
        self.beta = [np.zeros((self.k,)) for _ in range(self.level)]

        # wavelet transform
        V, W = [sig], [np.array(np.nan)]
        for _ in range(self.max_depth):
            _V, _W = pywt.dwt(V[-1], self.dwt_type)
            V.append(_V)
            W.append(_W)

        # pruning
        self.W = [np.array(np.nan)] + [
            Wi[-max(self.n_ls)-1:] for Wi in W[1:]
        ]
        self.V = [Vi[-max(self.n_ls)-1:] for Vi in V]

        # prepare for variables
        for l in range(self.level):
            X, y = list(), list()
            for t in range(max(self.n_ls), W[l+1].shape[0]):
                _X = list()
                for d in range(self.depth):
                    _t = t // (2**d)
                    _X.append(W[l+d+1][_t-self.n_ls[d]: _t])
                _X = np.concatenate(_X)
                if _X.shape[0] == self.k:
                    X.append(_X)
                    y.append(W[l+1][t])
            if X:
                X, y = np.array(X), np.array(y)
                self.P[l], self.q[l] = X.T @ X, X.T @ y
                try:
                    self.beta[l] = np.linalg.inv(self.P[l]) @ self.q[l]
                except np.linalg.LinAlgError:
                    self.beta[l] = np.linalg.pinv(self.P[l]) @ self.q[l]
            else:
                break

    def update_crest(self, x_t):
        updated_W_levels = list()
        # TODO: "db6"にも対応させる必要あり
        for l in range(self.max_depth - 1):
            if (self.current_t + 1) % (2**l) == 0:
                self.__compute_V(l, x_t)
            if (self.current_t + 1) % (2 ** (l + 1)) == 0:
                self.__compute_W(l + 1)
                updated_W_levels.append(l + 1)
        self.current_t += 1

        return updated_W_levels

    def update(self, x_t):
        updated_W_levels = self.update_crest(x_t)
        for l in updated_W_levels:
            X = np.concatenate([
                Wi[-ni-1: -1]
                for Wi, ni in zip(self.W[l: l+self.depth], self.n_ls)
            ])
            y = self.W[l][-1]
            if X.shape[0] == self.k:
                self.P[l] += X @ X.T
                self.q[l] += y * X

    def model_selection(self):
        raise NotImplementedError

    def predict(self, st, interval):
        W = copy.deepcopy(self.W)
        W_preds = [np.array([])] * self.level
        for t in range(st, st + interval):
            for l in range(self.level):
                if t % (2 ** (l + 1)) == 0:
                    X = np.concatenate([
                        Wi[-ni:]
                        for Wi, ni in zip(W[l+1: l+self.depth+1], self.n_ls)
                    ])
                    if X.shape[0] == self.k:
                        W_pred = X @ self.beta[l]
                        W_preds[l] = np.append(W_preds[l], W_pred)
        W_preds = [pred for pred in W_preds if len(pred)]
        W_preds.append(np.zeros_like(W_preds[-1]) + max(self.V[-1]))
        W_preds = W_preds[::-1]
        return pywt.waverec(W_preds, self.dwt_type)

    def __compute_V(self, level, x_t):
        if level == 0:
            self.V[0] = self.__fifo(self.V[0], x_t)
        else:
            V_new = pywt.dwt(self.V[level - 1][-2:], self.dwt_type)[0]
            self.V[level] = self.__fifo(self.V[level], V_new)

    def __compute_W(self, level):
        W_new = pywt.dwt(self.V[level - 1][-2:], self.dwt_type)[1]
        self.W[level] = self.__fifo(self.W[level], W_new)

    def __fifo(self, datas, new_data):
        return np.delete(np.append(datas, new_data), 0)
