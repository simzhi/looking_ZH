import torch.nn as nn
import torch
from torchvision import models

class LookingModel(nn.Module):
    def __init__(self, input_size, output_size=1, linear_size=256, p_dropout=0.2, num_stage=3, bce=False):
        super(LookingModel, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.num_stage = num_stage
        self.bce = bce
        self.dropout = nn.Dropout(self.p_dropout)
        self.relu = nn.ReLU(inplace=True)

        # process input to linear size
        self.w1 = nn.Linear(self.input_size, self.linear_size)
        self.batch_norm1 = nn.BatchNorm1d(self.linear_size)

        self.linear_stages = []
        for _ in range(num_stage):
            self.linear_stages.append(Linear(self.linear_size, self.p_dropout))
        self.linear_stages = nn.ModuleList(self.linear_stages)

        # post processing
        self.w2 = nn.Linear(self.linear_size, self.output_size)
        self.sigmoid = nn.Sigmoid()
        #self.relu = nn.ReLU(inplace=True)
        #self.dropout = nn.Dropout(self.p_dropout)

    def forward(self, x):
        # pre-processing
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)
        # linear layers
        for i in range(self.num_stage):
            y = self.linear_stages[i](y)
        y = self.w2(y)
        #y = self.dropout(y)
        y = self.sigmoid(y)
        return y

class Linear(nn.Module):
    def __init__(self, linear_size=256, p_dropout=0.2):
        super(Linear, self).__init__()

        ###

        self.linear_size = linear_size
        self.p_dropout = p_dropout

        ###

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.p_dropout)

        ###
        self.l1 = nn.Linear(self.linear_size, self.linear_size)
        self.bn1 = nn.BatchNorm1d(self.linear_size)

        self.l2 = nn.Linear(self.linear_size, self.linear_size)
        self.bn2 = nn.BatchNorm1d(self.linear_size)
        
    def forward(self, x):
        # stage I

        y = self.l1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.dropout(y)

        # stage II

        
        y = self.l2(y)
        y = self.bn2(y)
        y = self.relu(y)
        y = self.dropout(y)

        # concatenation

        out = x+y

        return out


class LookingNet_early_fusion_sum(nn.Module):
    def __init__(self, PATH, PATH_look, device):
        super(LookingNet_early_fusion_sum, self).__init__()
        self.backbone = models.resnext50_32x4d(pretrained=False)
        self.backbone.fc  = nn.Sequential(
            nn.Linear(in_features=2048, out_features=1, bias=True),
            nn.Sigmoid()
        )
        self.backbone.load_state_dict(torch.load(PATH))
        for m in self.backbone.parameters():
            m.requires_grad = False
        self.backbone.eval()

        self.looking_model = LookingModel(51)
        self.looking_model = torch.load(PATH_look, map_location=torch.device(device))
        for m in self.looking_model.parameters():
            m.requires_grad = False
        self.looking_model.eval()



        self.encoder_head = nn.Sequential(
            nn.Linear(2048, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),
            nn.ReLU(inplace=True)
        )
        self.final = nn.Sequential(
            nn.Linear(256, 1, bias=False),
            nn.Sigmoid()
        )


    def forward(self, head, keypoint):
        
        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach().squeeze()
            return hook
        def get_activation2(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook

        self.backbone.avgpool.register_forward_hook(get_activation('avgpool'))
        self.looking_model.linear_stages[2].bn2.register_forward_hook(get_activation('look'))

        #x = torch.randn(1, 25)
        output_head = self.backbone(head)
        out_kps = self.looking_model(keypoint)

        layer_look = activation["look"]
        layer_resnet = activation["avgpool"]

        out_final = self.encoder_head(layer_resnet)+layer_look

        return self.final(out_final)

class LookingNet_early_fusion_18(nn.Module):
    def __init__(self, PATH, PATH_look, device):
        super(LookingNet_early_fusion_18, self).__init__()
        self.backbone = models.resnet18(pretrained=False)
        self.backbone.fc  = nn.Sequential(
            nn.Linear(in_features=512, out_features=1, bias=True),
            nn.Sigmoid()
        )
        self.backbone.load_state_dict(torch.load(PATH))
        for m in self.backbone.parameters():
            m.requires_grad = False
        self.backbone.eval()

        self.looking_model = LookingModel(51)
        self.looking_model = torch.load(PATH_look, map_location=torch.device(device))
        for m in self.looking_model.parameters():
            m.requires_grad = False
        self.looking_model.eval()



        self.encoder_head = nn.Sequential(
            nn.Linear(512, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.Dropout(0.4),
            nn.ReLU(inplace=True),
            nn.Linear(64, 16),
            nn.BatchNorm1d(16),
            nn.Dropout(0.4),
            nn.ReLU(inplace=True)
        )
        self.final = nn.Sequential(
            nn.Linear(272, 1, bias=False),
            nn.Sigmoid()
        )


    def forward(self, head, keypoint):
        
        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach().squeeze()
            return hook
        def get_activation2(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook

        self.backbone.avgpool.register_forward_hook(get_activation('avgpool'))
        self.looking_model.linear_stages[2].bn2.register_forward_hook(get_activation('look'))

        #x = torch.randn(1, 25)
        output_head = self.backbone(head)
        out_kps = self.looking_model(keypoint)

        layer_look = activation["look"]
        layer_resnet = activation["avgpool"]

        out_final = torch.cat((self.encoder_head(layer_resnet), layer_look), 1).type(torch.float)

        return self.final(out_final)

class LookingNet_early_fusion_50(nn.Module):
    def __init__(self, PATH, PATH_look, device):
        super(LookingNet_early_fusion_50, self).__init__()
        self.backbone = models.resnext50_32x4d(pretrained=False)
        self.backbone.fc  = nn.Sequential(
            nn.Linear(in_features=2048, out_features=1, bias=True),
            nn.Sigmoid()
        )
        self.backbone.load_state_dict(torch.load(PATH))
        for m in self.backbone.parameters():
            m.requires_grad = False
        self.backbone.eval()

        self.looking_model = LookingModel(51)
        self.looking_model = torch.load(PATH_look, map_location=torch.device(device))
        for m in self.looking_model.parameters():
            m.requires_grad = False
        self.looking_model.eval()



        self.encoder_head = nn.Sequential(
            nn.Linear(2048, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.Dropout(0.4),
            nn.ReLU(inplace=True),
            nn.Linear(64, 16),
            nn.BatchNorm1d(16),
            nn.Dropout(0.4),
            nn.ReLU(inplace=True)
        )
        self.final = nn.Sequential(
            nn.Linear(272, 1, bias=False),
            nn.Sigmoid()
        )


    def forward(self, head, keypoint):
        
        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach().squeeze()
            return hook
        def get_activation2(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook

        self.backbone.avgpool.register_forward_hook(get_activation('avgpool'))
        self.looking_model.linear_stages[2].bn2.register_forward_hook(get_activation('look'))

        #x = torch.randn(1, 25)
        output_head = self.backbone(head)
        out_kps = self.looking_model(keypoint)

        layer_look = activation["look"]
        layer_resnet = activation["avgpool"]

        out_final = torch.cat((self.encoder_head(layer_resnet), layer_look), 1).type(torch.float)

        return self.final(out_final)