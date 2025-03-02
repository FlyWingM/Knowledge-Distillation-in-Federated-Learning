### centralized_learning.py

import torch
import torch.nn as nn
from train_central.train_cen_model import Model_Cen_Trainer
from train_fl.evaluate_model import ModelEvaluator
from nn_architecture.nn_selection import nn_archiecture_selection
from torch.optim.lr_scheduler import CosineAnnealingLR

class CentralizedServer:

    def __init__(self, g_para, dataset):
        self.g_para = g_para
        self.g_para.numEpoch = self.g_para.l_iter_group[0]

        print(f"\n3. Creating the central MOC")
        self.cen_model = nn_archiecture_selection(self.g_para, self.g_para.device)  
        self.cen_model = torch.nn.DataParallel(self.cen_model).cuda()
        self.cen_optimizer = torch.optim.SGD(
            self.cen_model.parameters(), 
            lr=self.g_para.h_param['learning_rate'], 
            momentum=self.g_para.h_param['momentum'], 
            weight_decay=self.g_para.h_param['weight_decay']
        )
        self.cen_criterion = nn.CrossEntropyLoss()
        self.cen_scheduler = CosineAnnealingLR(
            optimizer=self.cen_optimizer, 
            T_max=self.g_para.numEpoch, 
            eta_min=0
        )

        self.central_train = dataset.dataloader_central_train
        self.central_valid = dataset.dataloader_central_valid
        self.central_test = dataset.dataloader_central_test


class FLParameters:
    def __init__(self, g_para, dataset, server, clients):
        self.g_para = g_para
        self.dataset = dataset
        self.server = server
        self.clients = clients

class CentralizedTraining:

    def __init__(self, g_para, dataset):
        self.g_para = g_para		
        self.dataset = dataset		
        self.server = CentralizedServer(self.g_para, self.dataset)
        self.clients = None
        self.fl_params =  FLParameters(self.g_para, self.dataset, self.server, self.clients)
        self.model_evaluator = ModelEvaluator(self.fl_params)
        self.model_train = Model_Cen_Trainer(self.fl_params)


    def train(self):
        print("\n5) Start to train the centralized model")
        self.model_train.train_cen_model()
        self.model_evaluator.cmodel_evaluate()


