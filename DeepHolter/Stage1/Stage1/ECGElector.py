import torch
import torch.nn as nn
class ECGElector(nn.Module):
    def __init__(self, num_selected=6, group_size=60):
        super().__init__()
        self.num_selected = num_selected
        self.group_size = group_size
        
        self.global_pool = nn.Sequential(

            nn.AdaptiveAvgPool1d(100),  # 2000 -> 100
            nn.Conv1d(12, 24, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool1d(1),  # 100 -> 1
            nn.Flatten(start_dim=1)  # (24,)
        )
        
        self.scorer = nn.Sequential(
            nn.Linear(24, 64),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, input_tensor, data_mask=None, bag_mask=None):
        """
        
        Args:
            input_tensor: (batch, 360, 24, 2000, 12)
            data_mask: (batch, 360, 24), 0 is pad
            
        Returns:
            selected_indices: (batch, 6)
        """
        batch_size, num_bags, num_instances, seq_len, num_leads = input_tensor.shape
        
        # 1. 提取每个instance的特征
        reshaped = input_tensor.reshape(-1, seq_len, num_leads)
        
        reshaped = reshaped.permute(0, 2, 1)
        
        instance_features = self.global_pool(reshaped)  # (batch*360*24, 24)
        
        instance_features = instance_features.reshape(
            batch_size, num_bags, num_instances, -1
        )
        
        # 2. 计算每个instance的分数
        instance_scores = self.scorer(instance_features)  # (batch, 360, 24, 1)
        instance_scores = instance_scores.squeeze(-1)  # (batch, 360, 24)
        
        if data_mask is not None:
            assert data_mask.shape == (batch_size, num_bags, num_instances), \
                f"data_mask shape {data_mask.shape} should be (batch, 360, 24)"
            
            instance_scores = instance_scores.masked_fill(data_mask == 0, -1e9)
        else:
            with torch.no_grad():

                instance_energy = torch.norm(input_tensor, dim=(3, 4))  # (batch, 360, 24)

                instance_has_signal = (instance_energy > 1e-6).float()
            

            instance_scores = instance_scores.masked_fill(instance_has_signal == 0, -1e9)
        

        bag_scores, _ = torch.max(instance_scores, dim=2)  # (batch, 360)
        
        if bag_mask is not None:
            assert bag_mask.shape == (batch_size, num_bags), \
                f"bag_mask shape {bag_mask.shape} should be (batch, 360)"
            bag_scores = bag_scores.masked_fill(bag_mask == 0, -1e9)

        if data_mask is not None:
            bag_has_valid = (data_mask.sum(dim=2) > 0).float()  # (batch, 360)
            bag_scores = bag_scores.masked_fill(bag_has_valid == 0, -1e9)
        else:

            bag_has_valid = (instance_has_signal.sum(dim=2) > 0).float()
            bag_scores = bag_scores.masked_fill(bag_has_valid == 0, -1e9)
        
        # 6. 分组选举
        selected_indices = self._elect_by_groups(bag_scores)
        
        return selected_indices
    
    def _elect_by_groups(self, bag_scores):
        """分组选举实现"""
        batch_size, num_bags = bag_scores.shape
        
        assert num_bags == self.num_selected * self.group_size, \
            f"Expected {self.num_selected * self.group_size} bags, got {num_bags}"
        
        group_scores = bag_scores.view(batch_size, self.num_selected, self.group_size)
        
        _, group_indices = torch.max(group_scores, dim=2)  # (batch, num_selected)
        
        offsets = torch.arange(0, num_bags, self.group_size, 
                             device=bag_scores.device).unsqueeze(0)  # (1, num_selected)
        global_indices = group_indices + offsets
        
        return global_indices