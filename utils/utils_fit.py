import os
from tqdm import tqdm
from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from .utils_metrics import ConfusionMatrix
from .utils import get_lr
import numpy as np



def fit_one_epoch(model_train, model, loss_history,optimizer,
                    epoch,epoch_step,epoch_step_val,
                    gen,gen_val,
                    Epoch,cuda,
                    save_period,num_classes):
    train_loss = 0
    train_accuracy = 0

    print("*****Start Train*****")
    pbar = tqdm(total = epoch_step, desc=f'Epoch{epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.train()
    for iteration,batch in enumerate(gen):
        images_US,images_TH,targets = batch

        with torch.no_grad():
            if cuda:
                images_US = images_US.cuda()
                images_TH = images_TH.cuda()
                targets = targets.cuda()
        
        optimizer.zero_grad()

        outputs = model_train(images_US,images_TH)
        

        loss_value = nn.CrossEntropyLoss()(outputs, targets)

        loss_value.backward()

        optimizer.step()

        train_loss += loss_value.item()
        with torch.no_grad():
            outputs = torch.argmax(F.softmax(outputs, dim=-1), dim=-1)
            accuracy = torch.mean((outputs == targets).type(torch.FloatTensor))
            train_accuracy += accuracy.item()
        
        pbar.set_postfix(**{
            'train_loss':train_loss/(iteration + 1),
            'accuracy':train_accuracy/(iteration + 1),
            'lr':get_lr(optimizer)
        })

        pbar.update(1)
    print("*****Finish Train*****")
    

    val_loss = 0
    val_accuracy = 0
    labels = range(num_classes)
    confusion = ConfusionMatrix(num_classes=num_classes,labels=labels)
    print("*****Start Validation*****")
    pbar = tqdm(total=epoch_step_val, desc=f'Epoch{epoch+1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.eval()
    with torch.no_grad():
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_step_val:
                break
            images_US,images_TH,targets = batch
            with torch.no_grad():
                if cuda:
                    images_US = images_US.cuda()
                    images_TH = images_TH.cuda()
                    targets = targets.cuda()

                outputs = model_train(images_US,images_TH)
            
            loss_value = nn.CrossEntropyLoss()(outputs,targets)
            val_loss += loss_value.item()
            outputs = torch.argmax(F.softmax(outputs,dim=-1),dim=-1)
            confusion.update(outputs.to("cpu").detach().numpy(), targets.to("cpu").detach().numpy())
            accuracy = torch.mean((outputs == targets).type(torch.FloatTensor))
            val_accuracy += accuracy.item()
            pbar.set_postfix(**{
                'val_loss':val_loss / (iteration + 1),
                'accuracy':val_accuracy / (iteration + 1),
                'lr':get_lr(optimizer)
            })
            pbar.update(1)
    print("*****Finish Validation*****")
    
    
    precisions,recalls,specificitys,f1s,val_acc,_,_,_,_ = confusion.summary(is_print=False)
    
    loss_history.append_metrics(epoch+1, confusion.matrix, round(train_loss / epoch_step,4), round(val_loss / epoch_step_val,4), round(train_accuracy / epoch_step,4), round(val_acc,4), precisions, recalls, specificitys, f1s)
    print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    print('Train Loss: %.3f || Val Loss: %.3f' % (train_loss / epoch_step, val_loss / epoch_step_val) )
    print('Train Acc: %.3f || Val Acc: %.3f' % (train_accuracy / epoch_step, val_accuracy / epoch_step_val) )

    if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
        torch.save(model.state_dict(), os.path.join(loss_history.log_dir,"ep%03d-loss%.3f-val_loss%.3f-acc%.3f-val_acc%.3f.pth" % (epoch + 1, train_loss / epoch_step, val_loss / epoch_step_val,train_accuracy / epoch_step, val_accuracy / epoch_step_val)))
    
    if len(loss_history.val_metrics["loss"]) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_metrics["loss"]):
        print('Save best model to best_epoch_weights.pth')
        torch.save(model.state_dict(),os.path.join(loss_history.log_dir,"best_epoch_weights.pth"))
    
    torch.save(model.state_dict(), os.path.join(loss_history.log_dir,"last_epoch_weights.pth"))










