import shutil
import torch.nn.parallel
from options import Options
from dataloader import MyDataset
from utils import Bar, Logger, AverageMeter
from model import *
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
from torch.utils import data
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import time


def main():
    import warnings
    warnings.filterwarnings("ignore")  # ignore resnet warning
    opt = Options().parse()
    transform_v = transforms.Compose([transforms.Resize([opt.cropWidth, opt.cropHeight]),  
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transform_t = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    trainset = MyDataset('data', opt.length, transform_v, transform_t, 'train')
    train_loader = torch.utils.data.DataLoader(
        dataset=trainset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers)
    )
    testset = MyDataset('data', opt.length, transform_v, transform_t, 'test')
    test_loader = torch.utils.data.DataLoader(
        dataset=testset,
        batch_size=opt.batchSize,
        shuffle=False,
        num_workers=int(opt.workers)
    )

    #model = resnet_lstm()
    #model = cnn_tactile_lstm()
    #model = v_resnet_tcn()
    #model = t_cnn_tcn()

    #model = vt_resnet_cnn_lstm()

    model = vt_resnet_tcn_tcn()

    if opt.use_cuda:
        model = torch.nn.DataParallel(model).cuda()
        print("gpu is used!!!!")
    else:
        model.to(torch.device('cpu'))
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    params_conv = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(params_conv, lr=opt.lr)

    title = opt.name
    logger = Logger(os.path.join(opt.checkpoint, 'log.txt'), title=title)
    logger.set_names(['Learning Rate', 'Train Acc', 'Train Loss', 'Valid Loss', 'Valid Acc'])

    best_acc = 0
    train_acc_list = []
    train_loss_list = []
    test_acc_list = []
    test_loss_list = []
    for epoch in range(0, opt.epochs):
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, opt.epochs, lr))
   
        train_loss, train_acc = train(train_loader, model, optimizer, opt.use_cuda)
        test_loss, test_acc = test(test_loader, model, opt.use_cuda)

        # append logger file
        logger.append([opt.lr, train_acc, train_loss, test_loss, test_acc])
        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)
        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'acc': test_acc,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best, checkpoint=opt.checkpoint)
        print('Best acc:')
        print(best_acc)

    logger.close()
    opt.model_arch = 'cnn_mstcn'
    np.save(
        'XELA_results/train_acc_' + opt.model_arch + str(opt.batchSize) + '_' + str(opt.lr) + '.npy',
        train_acc_list)
    np.save('XELA_results/train_loss_' + opt.model_arch + str(opt.batchSize) + '_' + str(opt.lr) + '.npy',
            train_loss_list)
    np.save(
        'XELA_results/test_acc_' + opt.model_arch + str(opt.batchSize) + '_' + str(opt.lr) + '.npy',
        test_acc_list)
    np.save(
        'XELA_results/test_loss_' + opt.model_arch + str(opt.batchSize) + '_' + str(opt.lr) + '.npy',
        test_loss_list)
    plt.plot(train_loss_list)
    plt.plot(train_acc_list)
    plt.plot(test_loss_list)
    plt.plot(test_acc_list)

    plt.show()

def train(trainloader, model, optimizer, use_cuda):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    avg_acc = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(trainloader))
    for batch_idx, (x_visual, x_tactile, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        x_tactile, x_visual, targets = torch.autograd.Variable(x_tactile), \
                                       torch.autograd.Variable(x_visual), \
                                       torch.autograd.Variable(targets)

        if use_cuda:
            x_tactile = x_tactile.cuda()
            x_visual = x_visual.cuda()
            targets = targets.cuda(non_blocking=True)

        # compute output
        outputs = model(x_visual, x_tactile)
        #outputs = model(x_visual)
        #outputs = model(x_tactile)

        loss = F.cross_entropy(outputs, targets, reduction='mean')
        y_pred = torch.max(outputs, 1)[1]
        acc = accuracy_score(y_pred.cpu().data.numpy(), targets.cpu().data.numpy())

        # measure the result
        losses.update(loss.item(), x_tactile.size(0))
        avg_acc.update(acc, x_tactile.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress | PSNR: {psnr: .4f} | PSNR(input): {psnr_in: .4f}
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f}| ACC(input): {acc: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    acc=avg_acc.avg,
                    )
        bar.next()
    bar.finish()
    return losses.avg, avg_acc.avg


def test(testloader, model, use_cuda):
    # switch to train mode
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    avg_acc = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(testloader))
    for batch_idx, (x_visual, x_tactile, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        x_tactile, x_visual, targets = torch.autograd.Variable(x_tactile), \
                                       torch.autograd.Variable(x_visual), \
                                       torch.autograd.Variable(targets)
        if use_cuda:
            x_tactile = x_tactile.cuda()
            x_visual = x_visual.cuda()
            targets = targets.cuda(non_blocking=True)

        # compute output
        outputs = model(x_visual, x_tactile)
        #outputs = model(x_visual)
        #outputs = model(x_tactile)

        loss = F.cross_entropy(outputs, targets)
        y_pred = torch.max(outputs, 1)[1]
        acc = accuracy_score(y_pred.cpu().data.numpy(), targets.cpu().data.numpy())

        # measure the result
        losses.update(loss.item(), x_tactile.size(0))
        avg_acc.update(acc, x_tactile.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress | PSNR: {psnr: .4f} | PSNR(input): {psnr_in: .4f}
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f}| ACC(input): {acc: .4f}'.format(
            batch=batch_idx + 1,
            size=len(testloader),
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


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth'))


if __name__ == "__main__":
    main()
