import torch
from model import DM2FNet, DM2FNet_woPhy

model_R_Base = DM2FNet()
model_R_Improve = DM2FNet()
model_O_Base = DM2FNet_woPhy()
model_O_Improve = DM2FNet_woPhy()

model_R_Base.load_state_dict(torch.load('model/RESIDE_ITS_baseline/new/iter_40000_loss_0.01225_lr_0.000000.pth'))
model_R_Improve.load_state_dict(torch.load('model/RESIDE_ITS_improve/iter_40000_loss_0.01237_lr_0.000000.pth'))
model_O_Base.load_state_dict(torch.load('model/O-Haze_baseline/iter_20000_loss_0.05028_lr_0.000000.pth'))
model_O_Improve.load_state_dict(torch.load('model/O-Haze_improve/iter_20000_loss_0.04962_lr_0.000000.pth'))

# 计算总参数量和可训练参数量
total_params1 = sum(p.numel() for p in model_R_Base.parameters())
total_params2 = sum(p.numel() for p in model_R_Improve.parameters())
total_params3 = sum(p.numel() for p in model_O_Base.parameters())
total_params4 = sum(p.numel() for p in model_O_Improve.parameters())
# trainable_params = sum(p.numel() for p in model1.parameters() if p.requires_grad)

print(f"Total parameters for R_Base: {total_params1}")
print(f"Total parameters for R_Improve: {total_params2}")
print(f"Total parameters for O_Base: {total_params3}")
print(f"Total parameters for O_Improve: {total_params4}")
# print(f"Trainable parameters: {trainable_params}")
