import numpy as np
import torch.nn as nn
import torch
import vq_amm
from r_amm import im2col_transform

##
class MLP_Mixer_Manual():
    def __init__(self, model, N_SUBSPACE, K_CLUSTER):
        self.n_subspace = N_SUBSPACE
        self.k_cluster = K_CLUSTER
        self.to_patch_embedding = model.to_patch_embedding
        self.patch_weight = self.get_param(model.to_patch_embedding[0])
        self.mixer_blocks = model.mixer_blocks
        self.layer_norm = model.layer_norm
        self.mlp_weights = self.get_param(model.mlp_head)
        self.depth = len(model.mixer_blocks)
        self.sigmoid = nn.Sigmoid()

        self.amm_estimators = []
        self.amm_est_queue=[]
        self.gelu = nn.GELU()
        self.mm_res = []
        self.layer_res = []
    def get_param(self, model_layer, param_type="parameters"):
        if param_type == "parameters":
            return [param.detach().numpy() for param in model_layer.parameters()]
        elif param_type == "buffer":  # extract BN mean and var, remove track_num in buffer
            return [param.detach().numpy() for param in model_layer.buffers() if param.numel() > 1]
        else:
            raise ValueError("Invalid type in model layer to get parameters")

    def append_layer_res(self, val):
        if isinstance(val, torch.Tensor):
            val = val.detach().numpy()
        self.layer_res.append(val)

    def vector_matrix_multiply(self, vector, weight_t, bias=0, mm_type='exact'):
        # apply to batches of 1d and 2d vectors times weight matrix
        # (b;d)*(d,o)=>(b;o)
        # (b;n,d)*(d,o)=>(b;n,o)
        if isinstance(vector, torch.Tensor):
            vector = vector.detach().numpy()
        vec_shape = vector.shape
        if len(vec_shape)>2:
            vector = vector.reshape(-1,vec_shape[-1])

        if mm_type == 'exact':
            res = np.dot(vector, weight_t)
            if len(vec_shape) > 2:
                res = res.reshape(vec_shape[0],vec_shape[1],-1)
            res += bias
        elif mm_type == 'train_amm':
            ncodebooks,ncentroids = self.n_subspace.pop(0), self.k_cluster.pop(0)
            est = vq_amm.PQMatmul(ncodebooks, ncentroids)
            est.fit(vector, weight_t)
            est.reset_for_new_task()
            est.set_B(weight_t)
            res = est.predict(vector, weight_t)
            if len(vec_shape) > 2:
                res = res.reshape(vec_shape[0],vec_shape[1],-1)
            res += bias
            self.amm_estimators.append(est)
        elif mm_type == 'eval_amm':
            est = self.amm_est_queue.pop(0)
            est.reset_for_new_task() #necessary
            est.set_B(weight_t)
            res = est.predict(vector, weight_t)
            if len(vec_shape) > 2:
                res = res.reshape(vec_shape[0],vec_shape[1],-1)
            res += bias
        else:
            print("wrong mm_type!")
            return None
        self.mm_res.append(res)
        return res

    def token_mix_manual(self,input_data,layer,mm_type):
        x = layer[0:2](input_data)
        # MM1
        linear_1_param=self.get_param(layer[2].net[0])
        x = self.vector_matrix_multiply(x.detach().numpy(), linear_1_param[0].transpose(), linear_1_param[1], mm_type)
        x = self.gelu(torch.tensor(x)).detach().numpy()
        # MM2
        linear_2_param = self.get_param(layer[2].net[3])
        x = self.vector_matrix_multiply(x, linear_2_param[0].transpose(), linear_2_param[1], mm_type)

        x=layer[3](torch.tensor(x))
        return x

    def channel_mix_manual(self,input_data,layer,mm_type):
        x = layer[0](input_data)
        # MM1
        linear_1_param=self.get_param(layer[1].net[0])
        x = self.vector_matrix_multiply(x.detach().numpy(), linear_1_param[0].transpose(), linear_1_param[1], mm_type)
        x = self.gelu(torch.tensor(x)).detach().numpy()
        # MM2
        linear_2_param = self.get_param(layer[1].net[3])
        x = self.vector_matrix_multiply(x, linear_2_param[0].transpose(), linear_2_param[1], mm_type)
        x=torch.tensor(x)
        return x

    def patch_embedding_manual(self,input_data,weights,bias, mm_type='exact'):
        s=weights.shape[-1]
        col_matrix, kernel_reshaped, output_shape = im2col_transform(input_data, weights,s,padding=0,stride=s)
        #conv_result = np.dot(col_matrix, kernel_reshaped.transpose())
        conv_result = self.vector_matrix_multiply(col_matrix,kernel_reshaped.transpose(),0,mm_type)
        output = conv_result.reshape(*output_shape).transpose(0, 3, 1, 2)
        x = output + bias.reshape(1, -1, 1, 1)
        x = self.to_patch_embedding[1](torch.tensor(x, dtype=torch.float32))
        return x



    def forward(self,input_data, mm_type='exact'):
        self.layer_res.clear()
        self.mm_res.clear()

        #Patch Embedding/CNN: x = self.to_patch_embedding(input_data)
        x = self.patch_embedding_manual(input_data,self.patch_weight[0], self.patch_weight[1])
        self.append_layer_res(x)
        for i in range(self.depth):
            x = x + self.token_mix_manual(x, self.mixer_blocks[i].token_mix,mm_type)
            x = x + self.channel_mix_manual(x, self.mixer_blocks[i].channel_mix, mm_type)
            self.append_layer_res(x)

        x = self.layer_norm(x)
        x = x.mean(dim=1)

        #MLP: x=mlp_head(x)
        x = self.vector_matrix_multiply(x, self.mlp_weights[0].transpose(), self.mlp_weights[1], mm_type)
        x= self.sigmoid(torch.tensor(x))
        self.append_layer_res(x)

        mm_res = self.mm_res.copy()
        return self.layer_res, mm_res

    def forward_exact(self,input_data):
        return self.forward(input_data, mm_type='exact')

    def train_amm(self,input_data):
        output = self.forward(input_data, mm_type='train_amm')
        self.amm_est_queue = self.amm_estimators.copy()
        return output

    def eval_amm(self,input_data):
        return self.forward(input_data, mm_type='eval_amm')
