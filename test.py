# -*- coding: utf-8 -*-
import torch.nn.parallel
from options import Options
from dataloader import MyDataset
from utils import Bar, Logger, AverageMeter
import torchvision.transforms as transforms
from model import *
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
from torch.utils import data
import time

def main():
    import warnings
    warnings.filterwarnings("ignore")  # ignore resnet warning
    opt = Options().parse()
    model_path = r'result\vt_resnet_tcn_tcn\model_best.pth'
    transform_v = transforms.Compose([transforms.Resize([opt.cropWidth, opt.cropHeight]),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transform_t = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    validset = MyDataset('data', opt.length, transform_v, transform_t, 'test')
    val_loader = torch.utils.data.DataLoader(
        dataset=validset,
        batch_size=opt.batchSize,
        shuffle=False,
        num_workers=int(opt.workers)
    )
    #model = vt_resnet_cnn_lstm()
    #model = vt_resnet_tcn()

    model = vt_resnet_tcn_tcn()
    #model = v_resnet_tcn()
    #model = t_cnn_tcn()

    net_params_state_dict = torch.load(model_path)
    if opt.use_cuda:
        model = torch.nn.DataParallel(model).cuda()
        print("gpu is used!!!!")
    else:
        model.to(torch.device('cpu'))
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    model.load_state_dict(net_params_state_dict["state_dict"])

    best_acc = 0
    test_acc_list = []
    test_loss_list = []
    for epoch in range(1):  # （0,100）
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, 1, opt.lr))
        test_loss, test_acc = test(val_loader, model, opt.use_cuda)

        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)
        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        print('Best acc:')
        print(best_acc)


def test(val_loader, model, use_cuda):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    avg_acc = AverageMeter()
    end = time.time()
    bar = Bar('Processing', max=len(val_loader))

    for batch_idx, (x_visual, x_tactile, targets) in enumerate(val_loader):
        data_time.update(time.time() - end)
        x_tactile, x_visual, targets = torch.autograd.Variable(x_tactile), \
                                       torch.autograd.Variable(x_visual), \
                                       torch.autograd.Variable(targets)
        if use_cuda:
            x_tactile = x_tactile.cuda()
            x_visual = x_visual.cuda()
            targets = targets.cuda(non_blocking=True)
        outputs = model(x_visual, x_tactile)
        #outputs = model(x_visual)
        #outputs = model(x_tactile)
        loss = F.cross_entropy(outputs, targets)
        y_pred = torch.max(outputs, 1)[1]  # y_pred != output
        acc = accuracy_score(y_pred.cpu().data.numpy(), targets.cpu().data.numpy())
        losses.update(loss.item(), x_tactile.size(0))
        avg_acc.update(acc, x_tactile.size(0))


        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress | PSNR: {psnr: .4f} | PSNR(input): {psnr_in: .4f}
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f}| ACC(input): {acc: .4f}'.format(
            batch=batch_idx + 1,
            size=len(val_loader),
            data=data_time.avg,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            acc=avg_acc.avg,
        )
        bar.next()
    bar.finish()

    return (losses.avg, avg_acc.avg)
if __name__ == "__main__":
    main()
