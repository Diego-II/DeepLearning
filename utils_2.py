'''Basado en https://github.com/dccuchile/CC6204/blob/master/2020/tareas/tarea4/utils.py
'''
import sys
from timeit import default_timer as timer

import torch
from torch.utils.data import Dataset
import numpy as np

def train_for_classification2(net, train_loader, test_loader, optimizer,
                              criterion, lr_scheduler=None,
                              epochs=1, reports_every=1, device='cuda'):
    '''Para entrenar una red sin el diccionario raro a la salida.
    '''
    net.to(device)
    total_train = len(train_loader.dataset)
    total_test = len(test_loader.dataset)
    tiempo_epochs = 0
    train_loss, train_acc, test_acc = [], [], []

    for e in range(1, epochs+1):
        inicio_epoch = timer()

        # Aseguramos que todos los parámetros se entrenarán usando .train()
        net.train()

        # Variables para las métricas
        running_loss, running_acc = 0.0, 0.0

        for i, data in enumerate(train_loader):
            # Desagregamos los datos y los pasamos a la GPU
            X, Y = data
            X, Y = X.to(device), Y.to(device)

            # Limpiamos los gradientes, pasamos el input por la red, calculamos
            # la loss, ejecutamos el backpropagation (.backward)
            # y un paso del optimizador para modificar los parámetros
            optimizer.zero_grad()

            out_dict = net(X)
            # Y_logits = out_dict['logits']
            loss = criterion(out_dict, Y)

            loss.backward()
            optimizer.step()

            # loss
            items = min(total_train, (i+1) * train_loader.batch_size)
            running_loss += loss.item()
            avg_loss = running_loss/(i+1)

            # accuracy
            _, max_idx = torch.max(out_dict, dim=1)
            running_acc += torch.sum(max_idx == Y).item()
            avg_acc = running_acc/items*100

            # report
            sys.stdout.write(f'\rEpoch:{e}({items}/{total_train}), '
                             + (f'lr:{lr_scheduler.get_last_lr()[0]:02.7f}, ' if lr_scheduler is not None else '')
                             + f'Loss:{avg_loss:02.5f}, '
                             + f'Train Acc:{avg_acc:02.1f}%')

        tiempo_epochs += timer() - inicio_epoch

        if e % reports_every == 0:
            sys.stdout.write(', Validating...')
            train_loss.append(avg_loss)
            train_acc.append(avg_acc)
            net.eval()
            running_acc = 0.0
            for i, data in enumerate(test_loader):
                X, Y = data
                X, Y = X.to(device), Y.to(device)
                Y_logits = net(X)
                _, max_idx = torch.max(Y_logits, dim=1)
                running_acc += torch.sum(max_idx == Y).item()
                avg_acc = running_acc/total_test*100
            test_acc.append(avg_acc)
            sys.stdout.write(f', Val Acc:{avg_acc:02.2f}%, '
                             + f'Avg-Time:{tiempo_epochs/e:.3f}s.\n')
        else:
            sys.stdout.write('\n')

        if lr_scheduler is not None:
            lr_scheduler.step()

    return train_loss, (train_acc, test_acc)
