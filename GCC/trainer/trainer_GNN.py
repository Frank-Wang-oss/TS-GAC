import os
import sys

sys.path.append("..")
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.loss_GNN import NTXentLoss



def Trainer(model, temporal_contr_model, model_optimizer, temp_cont_optimizer, train_dl, test_dl, device, logger, config, experiment_log_dir, training_mode, lambda1, lambda2, lambda3,
            num_remain_aug1, num_remain_aug2):
    # Start training
    logger.debug("Training started ....")

    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, 'min')
    loss = []
    for epoch in range(1, config.num_epoch + 1):
        # Train and validate
        train_loss, train_acc, train_loss_details = model_train(model, temporal_contr_model, model_optimizer, temp_cont_optimizer, criterion, train_dl, config, device, training_mode, lambda1, lambda2, lambda3,
                                                                num_remain_aug1, num_remain_aug2)
        valid_loss, valid_acc, _, _ = model_evaluate(model, temporal_contr_model, test_dl, device, training_mode)
        if training_mode != 'self_supervised':  # use scheduler in all other modes.
            scheduler.step(valid_loss)
        loss.append(train_loss_details)
        logger.debug(f'\nEpoch : {epoch}\n'
                     f'Train Loss     : {train_loss:.4f}\t | \tTrain Accuracy     : {train_acc:2.4f}\n'
                     f'Valid Loss     : {valid_loss:.4f}\t | \tValid Accuracy     : {valid_acc:2.4f}')

    os.makedirs(os.path.join(experiment_log_dir, "saved_models"), exist_ok=True)
    chkpoint = {'model_state_dict': model.state_dict(), 'temporal_contr_model_state_dict': temporal_contr_model.state_dict()}
    torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", f'ckp_last.pt'))

    os.makedirs(os.path.join(experiment_log_dir, "saved_loss"), exist_ok=True)
    np.save(os.path.join(experiment_log_dir, "saved_loss", 'loss_test.npy'), loss)
    if training_mode != "self_supervised":  # no need to run the evaluation for self-supervised mode.
        # evaluate on the test set
        logger.debug('\nEvaluate on the Test set:')
        test_loss, test_acc, _, _ = model_evaluate(model, temporal_contr_model, test_dl, device, training_mode)
        logger.debug(f'Test loss      :{test_loss:0.4f}\t | Test Accuracy      : {test_acc:0.4f}')

    logger.debug("\n################## Training is Done! #########################")


def model_train(model, temporal_contr_model, model_optimizer, temp_cont_optimizer, criterion, train_loader, config, device, training_mode, lambda1, lambda2, lambda3,
                num_remain_aug1, num_remain_aug2):
    total_loss = []
    total_acc = []
    model.train()
    temporal_contr_model.train()
    loss_TC_1 = []
    loss_TC_2 = []
    loss_CC_graph = []
    loss_TC_node = []
    for batch_idx, (data, labels, aug1, aug2) in enumerate(train_loader):
        # send to device
        data, labels = data.float().to(device), labels.long().to(device)
        aug1, aug2 = aug1.float().to(device), aug2.float().to(device)

        # optimizer
        model_optimizer.zero_grad()
        temp_cont_optimizer.zero_grad()



        if training_mode == "self_supervised":
            predictions1, features1 = model(aug1, self_supervised = True, num_remain = num_remain_aug1)
            predictions2, features2 = model(aug2, self_supervised = True, num_remain = num_remain_aug2)

            # normalize projection feature vectors
            features1 = F.normalize(features1, dim=-1)
            features2 = F.normalize(features2, dim=-1)

            temp_cont_loss1, temp_cont_lstm_feat1 = temporal_contr_model(features1, features2)
            temp_cont_loss2, temp_cont_lstm_feat2 = temporal_contr_model(features2, features1)

            # normalize projection feature vectors
            zis = temp_cont_lstm_feat1 
            zjs = temp_cont_lstm_feat2 

        else:
            output = model(data)

        # compute loss
        if training_mode == "self_supervised":
            # lambda1 = 1
            # lambda2 = 0.7
            # lambda3 = 0.5
            nt_xent_criterion = NTXentLoss(device, config.batch_size, config.Context_Cont.temperature,
                                           config.Context_Cont.use_cosine_similarity)
            CC_graph, CC_node = nt_xent_criterion(zis, zjs)
            loss = (temp_cont_loss1 + temp_cont_loss2) * lambda1 + lambda2 * CC_graph + lambda3*CC_node
            # loss = (temp_cont_loss1 + temp_cont_loss2) * lambda1
            loss_TC_1.append(temp_cont_loss1.item())
            loss_TC_2.append(temp_cont_loss2.item())
            loss_CC_graph.append(CC_graph.item())
            loss_TC_node.append(CC_node.item())

        else: # supervised training or fine tuining
            predictions, features = output
            loss = criterion(predictions, labels)
            total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())

        total_loss.append(loss.item())
        loss.backward()
        model_optimizer.step()
        temp_cont_optimizer.step()

    total_loss = torch.tensor(total_loss).mean()

    if training_mode == "self_supervised":
        total_acc = 0
    else:
        total_acc = torch.tensor(total_acc).mean()
    return total_loss, total_acc, [np.mean(loss_TC_1), np.mean(loss_TC_2), np.mean(loss_CC_graph), np.mean(loss_TC_node)]
    # return total_loss, total_acc


def model_evaluate(model, temporal_contr_model, test_dl, device, training_mode):
    model.eval()
    temporal_contr_model.eval()

    total_loss = []
    total_acc = []

    criterion = nn.CrossEntropyLoss()
    outs = np.array([])
    trgs = np.array([])

    with torch.no_grad():
        for data, labels in test_dl:
            data, labels = data.float().to(device), labels.long().to(device)
            # print(labels)

            if training_mode == "self_supervised":
                pass
            else:
                output = model(data)

            # compute loss
            if training_mode != "self_supervised":
                predictions, features = output
                loss = criterion(predictions, labels)
                total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())
                total_loss.append(loss.item())

            if training_mode != "self_supervised":
                pred = predictions.max(1, keepdim=True)[1]  # get the index of the max log-probability
                outs = np.append(outs, pred.cpu().numpy())
                trgs = np.append(trgs, labels.data.cpu().numpy())

    if training_mode != "self_supervised":
        total_loss = torch.tensor(total_loss).mean()  # average loss
    else:
        total_loss = 0
    if training_mode == "self_supervised":
        total_acc = 0
        return total_loss, total_acc, [], []
    else:
        total_acc = torch.tensor(total_acc).mean()  # average acc

    return total_loss, total_acc, outs, trgs
