import torch
import os

from MiDaS.monodepth_net import MonoDepthNet

model_path = os.path.join("MiDaS", "model.pt")
model = MonoDepthNet(model_path)
sm = torch.jit.script(model)
sm.save("MiDaS/midas_mobile.pt")