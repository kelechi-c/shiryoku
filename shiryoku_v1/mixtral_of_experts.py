import torch
import torch.nn as nn 


class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.expert_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):

        return self.expert_net(x)


class RouterGate(nn.Module):
    def __init__(self, model_dim, n_experts):
        super().__init__()
        self.route_layer = nn.Linear(model_dim, n_experts)

    def forward(self, x):
        x = self.route_layer(x)
        routed_x = torch.nn.functional.softmax(x, dim=-1)

        return routed_x


class MixtralOfExpertsLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_experts):
        super().__init__()
        self.gate_network = RouterGate(input_dim, num_experts)
        self.experts = nn.ModuleList([Expert(input_dim, hidden_dim, output_dim) for _ in range(num_experts)])
        
    def forward(self, x, num_experts_chosen):
        
        gating_scores = self.gate_network(x)
        topk_gating_scores, topk_indices = gating_scores.topk(num_experts_chosen, dim=2, sorted=False)
        
        mask = torch.zeros_like(gating_scores).scatter_(2, topk_indices, 1)
        
        gating_scores = gating_scores * mask
        gating_scores = nn.functional.normalize(gating_scores, p=1, dim=2)
        
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=2)
        
        expert_outputs = expert_outputs.transpose(1, 2)
        output_logits = torch.einsum('bte,bteo->bto', gating_scores, expert_outputs)
        
        return output_logits 
