from functools import partial
import torch
import torch.nn as nn
from torch import optim
from dataset import FracNetTrainDataset,FracNetInferenceDataset
import transforms as tsfm
from metrics import dice, recall, precision, fbeta_score
from UNet import UNet
from lossfunction import SoftDiceLoss,MixLoss
from collections import OrderedDict
from tqdm import tqdm
import torchio
from lossfunction import SoftDiceLoss,MixLoss,FocalLoss

def val(model, val_loader,criterion):
    thresh = 0.1
    recall_partial = partial(recall, thresh=thresh)
    precision_partial = partial(precision, thresh=thresh)
    fbeta_score_partial = partial(fbeta_score, thresh=thresh)
    model.eval()
    val_dice = val_recall = val_precision = val_fbeta = loss = 0
    count = 0
    with torch.no_grad():
        for idx,(data, target) in tqdm(enumerate(val_loader),total=len(val_loader)):
            data, target = data.cuda(), target.cuda()
            output = model(data)
            loss  += ((criterion(output,target))*data.shape[0]).item()
            val_dice += (dice(output, target)*data.shape[0]).item()
            val_recall += (recall_partial(output,target)*data.shape[0]).item()
            val_precision +=  (precision_partial(output,target) *data.shape[0]).item()
            val_fbeta += (fbeta_score_partial(output,target) *data.shape[0]).item()
            count += data.shape[0]
        
    print('val_loss:',loss/count,'Val_dice',val_dice/count,'recall',val_recall/count,'precision',val_precision/count,'fbeta',val_fbeta/count)       
    val_log = OrderedDict({'Val_loss':loss/count, 'Val_dice': val_dice/count,'recall':val_recall/count,'precision':val_precision/count,'fbeta':val_fbeta/count})
    return val_log


def train(model, train_loader, optimizer,criterion):
    print("=======Epoch:{}=======lr:{}".format(epoch,optimizer.state_dict()['param_groups'][0]['lr']))
    model.train()
    count = 0
    train_loss = 0
    with tqdm(total=len(train_loader)) as t:
        for idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()
           
            optimizer.zero_grad()

            output = model(data)
            loss = criterion(output,target) 
            loss.backward()
            optimizer.step()
        
            train_loss = (train_loss*count + loss.item()*data.shape[0])/(count+data.shape[0])
            count += data.shape[0]
            t.set_postfix({'loss':train_loss})
            t.update(1)

    val_log = OrderedDict({'Train_Loss': train_loss})
    return val_log

if __name__ == '__main__':
    
    train_image_dir = 'train_image'
    train_label_dir = 'train_label'
    val_image_dir = 'val_image'
    val_label_dir = 'val_label'

    batch_size = 6
    num_workers = 6
    num_samples = 4

    

    model = UNet(hidden_channels = 16)
   
    model = nn.DataParallel(model.cuda())
    criterion = MixLoss(SoftDiceLoss(), 1, nn.BCEWithLogitsLoss(), 1)
    #criterion = MixLoss(SoftDiceLoss(), 1, FocalLoss(), 0.5)

    train_transforms = [
        tsfm.RandomFlip(),        #随机水平垂直翻转
        #tsfm.RandomRotate(),
        #tsfm.addNoise(),
        tsfm.Window(-200, 1000),
        tsfm.MinMaxNorm(-200, 1000)
        ]
    val_transforms = [
        tsfm.Window(-200, 1000),
        tsfm.MinMaxNorm(-200, 1000)
       
        ]
    ds_train = FracNetTrainDataset(train_image_dir, train_label_dir,num_samples=num_samples,
            transforms=train_transforms)
    dl_train = FracNetTrainDataset.get_dataloader(ds_train, batch_size, True,
            num_workers)
        
    ds_val = FracNetTrainDataset(val_image_dir, val_label_dir,num_samples=num_samples,train = False,
            transforms=val_transforms)
    dl_val = FracNetTrainDataset.get_dataloader(ds_val, batch_size, False,
            num_workers)
                           
    optimizer = optim.Adam(model.parameters(),lr = 1e-3)
    

    best = [0,100] 
    epoches = 50
   
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=30,gamma=0.1)
    for epoch in range(epoches):
    
        train_log = train(model, dl_train, optimizer,criterion)
        val_log = val(model, dl_val,criterion)

        if val_log['Val_loss'] < best[1]:
            print('Saving best model')
            torch.save(model.state_dict(),'best_model.pth')
            best[0] = epoch
            best[1] = val_log['Val_loss']
        print('Best performance at Epoch: {} | {}'.format(best[0],best[1]))
        lr_scheduler.step()
        torch.save(model.state_dict(),'models/epoch'+str(epoch)+'.pth')
        torch.cuda.empty_cache() 