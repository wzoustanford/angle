import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from mushroom_rl.utils.torch import get_weights, set_weights

import moore.utils.mixture_layers as mixture_layers

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class ActivationHook:
    """Pickleable hook class for storing activations"""
    def __init__(self, parent, name):
        self.parent = parent
        self.name = name
    
    def __call__(self, module, input, output):
        self.parent._h_activations[self.name] = output

class MetaworldSACMixtureMHCriticNetworkGRIN(nn.Module):
    def __init__(self, input_shape, 
                       output_shape, 
                       n_features,
                       activation = 'ReLU', 
                       n_head_features = [],
                       n_contexts = 1, 
                       subspace = None, 
                       orthogonal = False, 
                       n_experts = 4, 
                       agg_activation = ['ReLU', 'ReLU'], 
                       use_pretex_inhibition = False,
                       num_grin_recurrence = 1,
                       use_cuda = True, 
                       **kwargs):
        
        super().__init__()

        self._n_input = input_shape
        self._n_output = output_shape[0]

        self._use_cuda = use_cuda
        self._subspace = subspace
        self._n_contexts = n_contexts

        n_layers = len(n_features) #handle if the list is empty
        n_head_layers = len(n_head_features)

        self._task_encoder = nn.Linear(n_contexts, n_experts, bias = False)
        nn.init.xavier_uniform_(self._task_encoder.weight,
                                    gain=nn.init.calculate_gain('linear'))
        
        self._agg_activation = agg_activation

        self._h = nn.Sequential()
        
        input_size = self._n_input[0]

        if n_layers > 1:
            for i in range(0, n_layers):
                if i == n_layers - 1:
                    activation_fn = None
                    if not activation.lower() == "linear":
                        activation_fn = getattr(nn, activation)()
                    _activation = activation.lower()
                else:
                    activation_fn = nn.ReLU()
                    _activation = "relu"

                layer = nn.Linear(input_size, n_features[i])
                nn.init.xavier_uniform_(layer.weight,
                                gain=nn.init.calculate_gain(_activation))
                self._h.add_module(f"backbone_layer_{i}", layer)
                if activation_fn is not None:
                    self._h.add_module(f"act_{i}", activation_fn)
                
                input_size = n_features[i]


        if orthogonal:
            self._h = nn.Sequential(mixture_layers.InputLayer(n_models=n_experts),
                                    mixture_layers.ParallelLayer(self._h),
                                    mixture_layers.OrthogonalLayer1D(),
                                    )
        else:
            self._h = nn.Sequential(mixture_layers.InputLayer(n_models=n_experts),
                                    mixture_layers.ParallelLayer(self._h),
                                    )

        self.get_activation_list = ['_task_encoder']
        self.get_activation_list += [
            f'_h.1.model_layers.{str(i)}.act_0' for i in range(n_experts)
        ]
        self.get_activation_list += [
            f'_h.1.model_layers.{str(i)}.act_1' for i in range(n_experts)
        ]

        self._h_activations = dict()
        print('n_layers: ' + str(n_layers))
        self._hooks = []

        self._output_heads = nn.ModuleList()
        for c in range(n_contexts):
            head = nn.Sequential()

            input_size = n_features[-1]

            if n_head_layers > 0:
                for i in range(0, n_head_layers):
                    layer = nn.Linear(input_size, n_head_features[i])
                    nn.init.xavier_uniform_(layer.weight,
                                        gain=nn.init.calculate_gain('relu'))
                    head.add_module(f"head_{c}_layer_{i}",layer)

                    head.add_module(f"head_{c}_act_{i}",nn.ReLU())

                    input_size = n_head_features[i]
            
            layer = nn.Linear(input_size, self._n_output)
            nn.init.xavier_uniform_(layer.weight,
                                gain=nn.init.calculate_gain('linear'))
            head.add_module(f"head_{c}_out",layer)
            
            self._output_heads.append(head)
        
        #for i in range(n_contexts):
        #    self.get_activation_list.append(f"_output_heads.{i}.head_{i}_out")
        

        self.grin_inhibition_network = nn.Sequential(
            nn.Linear(n_experts + 2 * n_features[0] * n_experts + n_features[0] + self._n_output, 2 * n_features[0]),
            nn.Tanh(),
            nn.Linear(n_features[0] * 2, n_experts + n_features[0]),
            nn.Sigmoid(),
        )
        self.n_experts = n_experts 
        self.num_grin_recurrence = num_grin_recurrence 
        for name, modu in self.named_modules():
            print(name)
            if name in self.get_activation_list: #'_out' in name or ('act_' in name and '_h' in name):
                hook_handle = modu.register_forward_hook(self.get_activation(name))
                self._hooks.append(hook_handle)
    
    def get_activation(self, name):
        #def hook(module, input, output):
        #    self._h_activations[name] = output 
        return ActivationHook(self, name)

    def get_shared_weights_t(self):
        weights = []

        for l in self._h:
            if isinstance(l, nn.Linear):
                weights.append(l.weight)
                
        return weights
    
    def get_shared_weights(self):
        return [w.detach().cpu().numpy() for w in self.get_shared_weights_t()]

    def forward(self, state, action=None, c = None): 
        self._h_activations = dict()
        _ = self.forward_once(state, action, c, recurrent_pass=False)
        for i in range(self.num_grin_recurrence):
            q = self.forward_once(state, action, c, recurrent_pass=True)
        return q 

    def forward_once(self, state, action=None, c = None, recurrent_pass = False):
        # Clear activations at the start of each forward pass

        if isinstance(c, int):
            c = torch.tensor([c])

        if isinstance(c,np.ndarray):
            c = torch.from_numpy(c)

        if self._use_cuda:
            c = c.cuda()
        
        # task-weight computation
        c_onehot = F.one_hot(c, num_classes = self._n_contexts)
        w = self._task_encoder(c_onehot.float()).unsqueeze(1)

        state_action = torch.cat((state.float(), action.float()), dim=1)
        
        # shared features
        features = self._h(state_action)            

        features  = torch.permute(features, (1,0,2))

        # activation before
        if not self._agg_activation[0].lower() == "linear":
            features = getattr(torch, self._agg_activation[0].lower())(features)

        if recurrent_pass is True: 
            pretex_features = torch.Tensor().to(state.device)
            #pdb.set_trace()
            for name in self.get_activation_list:
                pretex_features = torch.cat((pretex_features, self._h_activations[name]), axis = 1)
            
            pretex_features = torch.cat((pretex_features, self._h_activations['features']), axis = 1)    
            pretex_features = torch.cat((pretex_features, self._h_activations['q']), axis = 1)

            inhibition_logits = self.grin_inhibition_network(pretex_features)
            w_inhibition_logits = inhibition_logits[:, :self.n_experts]
            features_inhibition_logits = inhibition_logits[:, self.n_experts:]
            w_inhibition_logits = w_inhibition_logits.unsqueeze(1)
            
            w = w * w_inhibition_logits 

        # task-features
        features = w@features
        features = features.squeeze(1)

        if recurrent_pass is True:
            features = features * features_inhibition_logits

        # activation after
        if not self._agg_activation[1].lower() == "linear":
            features = getattr(torch, self._agg_activation[1].lower())(features)
        
        self._h_activations['features'] = features

        q = torch.zeros(size=(state.shape[0], self._n_output))
        
        if self._use_cuda:
            q = q.cuda()

        for i, ci in enumerate(torch.unique(c)):
            ci_idx = torch.argwhere(c == ci).ravel()
            qi = self._output_heads[ci](features[ci_idx, :])
            q[ci_idx] = qi
        
        self._h_activations['q'] = q 
        return torch.squeeze(q)

class MetaworldSACMixtureMHActorNetworkGRIN(nn.Module):
    def __init__(self, input_shape, 
                       output_shape, 
                       n_features,
                       activation = 'ReLU', 
                       n_head_features = [],
                       shared_mu_sigma = False, 
                       n_contexts = 1,
                       subspace = None, 
                       orthogonal = False, 
                       n_experts = 4, 
                       agg_activation = ['ReLU', 'ReLU'], 
                       use_pretex_inhibition = False,
                       num_grin_recurrence = 1,
                       use_cuda = True, **kwargs):
        
        super().__init__()

        self._n_input = input_shape
        self._n_output = output_shape[0]

        if shared_mu_sigma:
            self._n_output*=2

        self._shared_mu_sigma = shared_mu_sigma

        self._use_cuda = use_cuda
        self._subspace = subspace
        self._n_contexts = n_contexts

        n_layers = len(n_features)
        n_head_layers = len(n_head_features)

        self._task_encoder = nn.Linear(n_contexts, n_experts, bias = False)
        nn.init.xavier_uniform_(self._task_encoder.weight,
                                    gain=nn.init.calculate_gain('linear'))
        
        self._agg_activation = agg_activation

        self._h = nn.Sequential()
        
        input_size = self._n_input[0]

        if n_layers > 1:
            for i in range(0, n_layers):
                if i == n_layers - 1:
                    activation_fn = None
                    if not activation.lower() == "linear":
                        activation_fn = getattr(nn, activation)()

                    _activation = activation.lower()
                else:
                    activation_fn = nn.ReLU()
                    _activation = "relu"

                layer = nn.Linear(input_size, n_features[i])
                nn.init.xavier_uniform_(layer.weight,
                                gain=nn.init.calculate_gain(_activation))
                self._h.add_module(f"backbone_layer_{i}", layer)
                if activation_fn is not None:
                    self._h.add_module(f"act_{i}", activation_fn)
                
                input_size = n_features[i]


        if orthogonal:
            self._h = nn.Sequential(mixture_layers.InputLayer(n_models=n_experts),
                                    mixture_layers.ParallelLayer(self._h),
                                    mixture_layers.OrthogonalLayer1D(),
                                    )
        else:
            self._h = nn.Sequential(mixture_layers.InputLayer(n_models=n_experts),
                                    mixture_layers.ParallelLayer(self._h),
                                    )
        
        self.get_activation_list = ['_task_encoder']
        self.get_activation_list += [
            f'_h.1.model_layers.{str(i)}.act_0' for i in range(n_experts)
        ]
        self.get_activation_list += [
            f'_h.1.model_layers.{str(i)}.act_1' for i in range(n_experts)
        ]
        
        self._h_activations = dict()
        print('n_layers: ' + str(n_layers))
        self._hooks = []

        self._output_heads = nn.ModuleList()
        for c in range(n_contexts):
            head = nn.Sequential()

            input_size = n_features[-1]

            if n_head_layers > 0:
                for i in range(0, n_head_layers):
                    layer = nn.Linear(input_size, n_head_features[i])
                    nn.init.xavier_uniform_(layer.weight,
                                        gain=nn.init.calculate_gain('relu'))
                    head.add_module(f"head_{c}_layer_{i}",layer)

                    head.add_module(f"head_{c}_act_{i}",nn.ReLU())

                    input_size = n_head_features[i]
            
            layer = nn.Linear(input_size, self._n_output)
            nn.init.xavier_uniform_(layer.weight,
                                gain=nn.init.calculate_gain('linear'))
            head.add_module(f"head_{c}_out",layer)
            
            self._output_heads.append(head)
        
        self.grin_inhibition_network = nn.Sequential(
            nn.Linear(n_experts + 2 * n_features[0] * n_experts + n_features[0] + self._n_output, 2*n_features[0]),
            nn.Tanh(),
            nn.Linear(n_features[0] * 2, n_experts + n_features[0]),
            nn.Sigmoid(),
        )
        self.n_experts = n_experts 
        self.num_grin_recurrence = num_grin_recurrence 
        for name, modu in self.named_modules():
            print(name)
            if name in self.get_activation_list: #'_out' in name or ('act_' in name and '_h' in name):
                hook_handle = modu.register_forward_hook(self.get_activation(name))
                self._hooks.append(hook_handle)

    def get_activation(self, name):
        #def hook(module, input, output):
        #    self._h_activations[name] = output 
        return ActivationHook(self, name)

    def get_shared_weights_t(self):
        weights = []

        for l in self._h:
            if isinstance(l, nn.Linear):
                weights.append(l.weight)
                
        return weights
    
    def get_shared_weights(self):
        return [w.detach().cpu().numpy() for w in self.get_shared_weights_t()]

    def forward(self, state, c = None): 
        self._h_activations = dict()
        _ = self.forward_once(state, c, recurrent_pass=False)
        for i in range(self.num_grin_recurrence):
            a = self.forward_once(state, c, recurrent_pass=True)
        return a 

    def forward_once(self, state, c = None, recurrent_pass = False):
        # Clear activations at the start of each forward pass

        if isinstance(c, int):
            c = torch.tensor([c])

        if isinstance(c,np.ndarray):
            c = torch.from_numpy(c)

        if self._use_cuda:
            c = c.cuda()
        
        # task-weight computation
        c_onehot = F.one_hot(c, num_classes = self._n_contexts)
        w = self._task_encoder(c_onehot.float()).unsqueeze(1)

        # shared features
        features = self._h(state.float())

        features  = torch.permute(features, (1,0,2))

        # activation before
        if not self._agg_activation[0].lower() == "linear":
            features = getattr(torch, self._agg_activation[0].lower())(features)

        if recurrent_pass is True: 
            pretex_features = torch.Tensor().to(state.device)
            #pdb.set_trace()
            for name in self.get_activation_list:
                pretex_features = torch.cat((pretex_features, self._h_activations[name]), axis = 1)
            
            pretex_features = torch.cat((pretex_features, self._h_activations['features']), axis = 1)    
            pretex_features = torch.cat((pretex_features, self._h_activations['a']), axis = 1)

            inhibition_logits = self.grin_inhibition_network(pretex_features)
            w_inhibition_logits = inhibition_logits[:, :self.n_experts]
            features_inhibition_logits = inhibition_logits[:, self.n_experts:]
            w_inhibition_logits = w_inhibition_logits.unsqueeze(1)
            
            w = w * w_inhibition_logits 

        # task-features
        features = w@features
        features = features.squeeze(1)

        if recurrent_pass is True:
            features = features * features_inhibition_logits

        # activation after
        if not self._agg_activation[1].lower() == "linear":
            features = getattr(torch, self._agg_activation[1].lower())(features)
        
        self._h_activations['features'] = features

        a = torch.zeros(size=(state.shape[0], self._n_output))
        
        if self._use_cuda:
            a = a.cuda()

        for i, ci in enumerate(torch.unique(c)):
            ci_idx = torch.argwhere(c == ci).ravel()
            ai = self._output_heads[ci](features[ci_idx, :])

            a[ci_idx] = ai
        
        self._h_activations['a'] = a 
        
        return a

class MetaworldSACMixtureMHCriticNetwork(nn.Module):
    def __init__(self, input_shape, 
                       output_shape, 
                       n_features,
                       activation = 'ReLU', 
                       n_head_features = [],
                       n_contexts = 1, 
                       subspace = None, 
                       orthogonal = False, 
                       n_experts = 4, 
                       agg_activation = ['ReLU', 'ReLU'], 
                       use_cuda = True, 
                       **kwargs):
        
        super().__init__()

        self._n_input = input_shape
        self._n_output = output_shape[0]

        self._use_cuda = use_cuda
        self._subspace = subspace
        self._n_contexts = n_contexts

        n_layers = len(n_features) #handle if the list is empty
        n_head_layers = len(n_head_features)

        self._task_encoder = nn.Linear(n_contexts, n_experts, bias = False)
        nn.init.xavier_uniform_(self._task_encoder.weight,
                                    gain=nn.init.calculate_gain('linear'))
        
        self._agg_activation = agg_activation


        self._h = nn.Sequential()
        
        input_size = self._n_input[0]

        if n_layers > 1:
            for i in range(0, n_layers):
                if i == n_layers - 1:
                    activation_fn = None
                    if not activation.lower() == "linear":
                        activation_fn = getattr(nn, activation)()

                    _activation = activation.lower()
                else:
                    activation_fn = nn.ReLU()
                    _activation = "relu"

                layer = nn.Linear(input_size, n_features[i])
                nn.init.xavier_uniform_(layer.weight,
                                gain=nn.init.calculate_gain(_activation))
                self._h.add_module(f"backbone_layer_{i}", layer)
                if activation_fn is not None:
                    self._h.add_module(f"act_{i}", activation_fn)
                
                input_size = n_features[i]


        if orthogonal:
            self._h = nn.Sequential(mixture_layers.InputLayer(n_models=n_experts),
                                    mixture_layers.ParallelLayer(self._h),
                                    mixture_layers.OrthogonalLayer1D(),
                                    )
        else:
            self._h = nn.Sequential(mixture_layers.InputLayer(n_models=n_experts),
                                    mixture_layers.ParallelLayer(self._h),
                                    )
    

        self._output_heads = nn.ModuleList()
        for c in range(n_contexts):
            head = nn.Sequential()

            input_size = n_features[-1]

            if n_head_layers > 0:
                for i in range(0, n_head_layers):
                    layer = nn.Linear(input_size, n_head_features[i])
                    nn.init.xavier_uniform_(layer.weight,
                                        gain=nn.init.calculate_gain('relu'))
                    head.add_module(f"head_{c}_layer_{i}",layer)

                    head.add_module(f"head_{c}_act_{i}",nn.ReLU())

                    input_size = n_head_features[i]
            
            layer = nn.Linear(input_size, self._n_output)
            nn.init.xavier_uniform_(layer.weight,
                                gain=nn.init.calculate_gain('linear'))
            head.add_module(f"head_{c}_out",layer)
            
            self._output_heads.append(head)

    def get_shared_weights_t(self):
        weights = []

        for l in self._h:
            if isinstance(l, nn.Linear):
                weights.append(l.weight)
                
        return weights
    
    def get_shared_weights(self):
        return [w.detach().cpu().numpy() for w in self.get_shared_weights_t()]

    def forward(self, state, action=None, c = None):
        if isinstance(c, int):
            c = torch.tensor([c])

        if isinstance(c,np.ndarray):
            c = torch.from_numpy(c)

        if self._use_cuda:
            c = c.cuda()
        
        # task-weight computation
        c_onehot = F.one_hot(c, num_classes = self._n_contexts)
        w = self._task_encoder(c_onehot.float()).unsqueeze(1)

        state_action = torch.cat((state.float(), action.float()), dim=1)
        
        # shared features
        features = self._h(state_action)
        features  = torch.permute(features, (1,0,2))

        # activation before
        if not self._agg_activation[0].lower() == "linear":
            features = getattr(torch, self._agg_activation[0].lower())(features)

        # task-features
        features = w@features
        features = features.squeeze(1)

        # activation after
        if not self._agg_activation[1].lower() == "linear":
            features = getattr(torch, self._agg_activation[1].lower())(features)

        q = torch.zeros(size=(state.shape[0], self._n_output))
        
        if self._use_cuda:
            q = q.cuda()

        for i, ci in enumerate(torch.unique(c)):
            ci_idx = torch.argwhere(c == ci).ravel()
            qi = self._output_heads[ci](features[ci_idx, :])
            q[ci_idx] = qi
            
        return torch.squeeze(q)

#######
#Actor#
#######      
class MetaworldSACMixtureMHActorNetwork(nn.Module):
    def __init__(self, input_shape, 
                       output_shape, 
                       n_features,
                       activation = 'ReLU', 
                       n_head_features = [],
                       shared_mu_sigma = False, 
                       n_contexts = 1,
                       subspace = None, 
                       orthogonal = False, 
                       n_experts = 4, 
                       agg_activation = ['ReLU', 'ReLU'], 
                       use_cuda = True, **kwargs):
        
        super().__init__()

        self._n_input = input_shape
        self._n_output = output_shape[0]

        if shared_mu_sigma:
            self._n_output*=2

        self._shared_mu_sigma = shared_mu_sigma

        self._use_cuda = use_cuda
        self._subspace = subspace
        self._n_contexts = n_contexts

        n_layers = len(n_features)
        n_head_layers = len(n_head_features)

        self._task_encoder = nn.Linear(n_contexts, n_experts, bias = False)
        nn.init.xavier_uniform_(self._task_encoder.weight,
                                    gain=nn.init.calculate_gain('linear'))
        
        self._agg_activation = agg_activation

        self._h = nn.Sequential()
        
        input_size = self._n_input[0]

        if n_layers > 1:
            for i in range(0, n_layers):
                if i == n_layers - 1:
                    activation_fn = None
                    if not activation.lower() == "linear":
                        activation_fn = getattr(nn, activation)()

                    _activation = activation.lower()
                else:
                    activation_fn = nn.ReLU()
                    _activation = "relu"

                layer = nn.Linear(input_size, n_features[i])
                nn.init.xavier_uniform_(layer.weight,
                                gain=nn.init.calculate_gain(_activation))
                self._h.add_module(f"backbone_layer_{i}", layer)
                if activation_fn is not None:
                    self._h.add_module(f"act_{i}", activation_fn)
                
                input_size = n_features[i]


        if orthogonal:
            self._h = nn.Sequential(mixture_layers.InputLayer(n_models=n_experts),
                                    mixture_layers.ParallelLayer(self._h),
                                    mixture_layers.OrthogonalLayer1D(),
                                    )
        else:
            self._h = nn.Sequential(mixture_layers.InputLayer(n_models=n_experts),
                                    mixture_layers.ParallelLayer(self._h),
                                    )
        
        self._output_heads = nn.ModuleList()
        for c in range(n_contexts):
            head = nn.Sequential()

            input_size = n_features[-1]

            if n_head_layers > 0:
                for i in range(0, n_head_layers):
                    layer = nn.Linear(input_size, n_head_features[i])
                    nn.init.xavier_uniform_(layer.weight,
                                        gain=nn.init.calculate_gain('relu'))
                    head.add_module(f"head_{c}_layer_{i}",layer)

                    head.add_module(f"head_{c}_act_{i}",nn.ReLU())

                    input_size = n_head_features[i]
            
            layer = nn.Linear(input_size, self._n_output)
            nn.init.xavier_uniform_(layer.weight,
                                gain=nn.init.calculate_gain('linear'))
            head.add_module(f"head_{c}_out",layer)
            
            self._output_heads.append(head)

    def get_shared_weights_t(self):
        weights = []

        for l in self._h:
            if isinstance(l, nn.Linear):
                weights.append(l.weight)
                
        return weights
    
    def get_shared_weights(self):
        return [w.detach().cpu().numpy() for w in self.get_shared_weights_t()]
    
    def forward(self, state, c = None):
        if isinstance(c, int):
            c = torch.tensor([c])

        if isinstance(c,np.ndarray):
            c = torch.from_numpy(c)

        if self._use_cuda:
            c = c.cuda()
        
        # task-weight computation
        c_onehot = F.one_hot(c, num_classes = self._n_contexts)
        w = self._task_encoder(c_onehot.float()).unsqueeze(1)

        # shared features
        features = self._h(state.float())
        features  = torch.permute(features, (1,0,2))

        # activation before
        if not self._agg_activation[0].lower() == "linear":
            features = getattr(torch, self._agg_activation[0].lower())(features)

        # task-features
        features = w@features
        features = features.squeeze(1)

        # activation after
        if not self._agg_activation[1].lower() == "linear":
            features = getattr(torch, self._agg_activation[1].lower())(features)

        a = torch.zeros(size=(state.shape[0], self._n_output))
        
        if self._use_cuda:
            a = a.cuda()

        for i, ci in enumerate(torch.unique(c)):
            ci_idx = torch.argwhere(c == ci).ravel()
            ai = self._output_heads[ci](features[ci_idx, :])

            a[ci_idx] = ai

        return a