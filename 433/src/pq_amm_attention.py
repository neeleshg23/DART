
import abc
import numpy as np

import vquantizers as vq
import product_quantize as pq
import amm

from vquantizers import (_learn_centroids,
                         MultiCodebookEncoder,
                         dists_elemwise_dot,
                         ensure_num_cols_multiple_of)


class PQ_ATT_Encoder(MultiCodebookEncoder):
    # PQ_Encoder: vquantizers.py
    def __init__(self, ncodebooks, ncentroids=16,
                 elemwise_dist_func=dists_elemwise_dot,
                 preproc='PQ', encode_algo=None, quantize_lut=False,
                 upcast_every=-1, accumulate_how='sum',
                 **preproc_kwargs):
        super().__init__(
            ncodebooks=ncodebooks, ncentroids=ncentroids,
            quantize_lut=quantize_lut, upcast_every=upcast_every,
            accumulate_how=accumulate_how)
        self.elemwise_dist_func = elemwise_dist_func
        self.preproc = preproc
        self.encode_algo = encode_algo
        self.preproc_kwargs = preproc_kwargs

    def _pad_ncols(self, X):
        return ensure_num_cols_multiple_of(X, self.ncodebooks)

    def fit(self, A, B):#kmeans learn centroids
        self.subvect_len = int(np.ceil(A.shape[1] / self.ncodebooks))
        A = self._pad_ncols(A)
        B = self._pad_ncols(B)
        self.centroids_A = None
        self.centroids_B = None
        if self.centroids_A is None:
            self.centroids_A = _learn_centroids(
                A, self.ncentroids, self.ncodebooks, self.subvect_len)
        if self.centroids_B is None:
            self.centroids_B = _learn_centroids(
                B, self.ncentroids, self.ncodebooks, self.subvect_len)
        return

    def encode_Q(self): #pairwise product

        A = self.centroids_A
        B = self.centroids_B

        luts = np.zeros((A.shape[0], B.shape[0], self.ncodebooks))
        # todo: udpate
        for i, a in enumerate(A):
            for j, b in enumerate(B):
                lut = np.sum(a * b, axis=-1)
                luts[i][j] = lut.T
        return luts

    def encode_X(self, X, centroids):#x->closest centroids->indexes
        X = self._pad_ncols(X)

        idxs = pq._encode_X_pq(X, codebooks=centroids)

        #return idxs + self.offsets  # offsets let us index into raveled dists

        return idxs

    def dists_enc_attn(self, A_enc, B_enc, luts):
        batch_size,d_a,n_codebooks = A_enc.shape
        d_b = B_enc.shape[1]

        all_res = np.empty((batch_size,n_codebooks,d_a,d_b), dtype=np.float32)

        for k in range(batch_size):

            for n in range(n_codebooks):
                a = A_enc[k,:,n]
                b = B_enc[k,:,n]
                a=[item for item in a for _ in range(d_b)]
                b=list(b)*d_a
                all_res[k,n]=luts[:,:,n][a,b].reshape(d_a,d_b)

        all_res = np.sum(all_res,axis=1)

        return all_res



class PQ_AMM_ATTENTION():
    # VQMatmul: vq_amm.py
    def __init__(self, ncodebooks, ncentroids=None):
        self.ncodebooks = ncodebooks
        self.ncentroids = ncentroids
        self.enc = self._create_encoder(ncodebooks)
        self.reset_for_new_task()

    def _create_encoder(self, ncodebooks):  # to be overriden by subclasses
        return PQ_ATT_Encoder(ncodebooks=ncodebooks, ncentroids=self.ncentroids,
                            **self._get_encoder_kwargs())

    def reset_enc(self): #A_enc: predicted centroid indexes
        self.A_enc = None
        self.B_enc = None

    def reset_for_new_task(self):
        self.A_enc = None
        self.B_enc = None
        self.luts = None


    def _get_encoder_kwargs(self):  # to be overriden by subclasses
        return {}

    def fit(self, A, B):
        _, X, D = A.shape
        if D < self.ncodebooks:
            raise amm.InvalidParametersException(
                'D < k: {} < {}'.format(D, self.ncodebooks))
        B=np.transpose(B, (0, 2, 1))
        self.enc.fit(A.reshape(-1,D), B.reshape(-1,D))

    def set_A(self, A, B):
        batch, step, D = A.shape
        A=A.reshape(-1,D)
        B=np.transpose(B, (0, 2, 1)).reshape(-1,D)
        self.A_enc = self.enc.encode_X(A, self.enc.centroids_A).reshape(batch,-1,self.ncodebooks)
        self.B_enc = self.enc.encode_X(B, self.enc.centroids_B).reshape(batch,-1,self.ncodebooks)

    def set_B(self):
        self.luts = self.enc.encode_Q()

    def predict(self, A, B):
        if self.A_enc is None:
            self.set_A(A,B)
        if self.luts is None:
            self.set_B()
        return self.enc.dists_enc_attn(self.A_enc, self.B_enc, self.luts)


##
if __name__ == "__main__":
    mat_1 = np.random.rand(1000, 11, 16)
    mat_2 = np.random.rand(1000, 16, 14)

    est = PQ_AMM_ATTENTION(4,256)
    est.fit(mat_1,mat_2)
    est.set_B()
    m1, m2 = mat_1[:500],mat_2[:500]

    est.set_A(m1, m2)
    res_amm = est.predict(m1,m2)
    res_exact = np.matmul(m1,m2)
    from metrics import _cossim
    cos=_cossim(res_amm, res_exact)
    print(cos)
    print("done")
    #0.9984333112131634