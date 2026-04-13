import os
import time
import random
import logging
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingLR

# Project-specific imports
from lib.networks import EMCADNet
from utils.dataloader_polyp import get_loader as get_loader
from utils.utils import clip_gradient, adjust_lr, AvgMeter, cal_params_flops


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def structure_loss(pred, mask, w=1):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (w * (wbce + wiou)).mean()

def dice_coefficient(predicted, labels):
    if predicted.device != labels.device:
        labels = labels.to(predicted.device)
    smooth = 1e-6
    predicted_flat = predicted.contiguous().view(-1)
    labels_flat = labels.contiguous().view(-1)
    intersection = (predicted_flat * labels_flat).sum()
    total = predicted_flat.sum() + labels_flat.sum()
    return (2. * intersection + smooth) / (total + smooth)

def iou(predicted, labels):
    if predicted.device != labels.device:
        labels = labels.to(predicted.device)
    smooth = 1e-6
    predicted_flat = predicted.contiguous().view(-1)
    labels_flat = labels.contiguous().view(-1)
    intersection = (predicted_flat * labels_flat).sum()
    union = predicted_flat.sum() + labels_flat.sum() - intersection
    return (intersection + smooth) / (union + smooth)


def adjacent_head_disagreement(predictions):
    if len(predictions) < 4:
        return 0.0
    probs = [torch.sigmoid(pred.detach()) for pred in predictions[:4]]
    disagreements = []
    for idx in range(3):
        dice = dice_coefficient(probs[idx], probs[idx + 1]).item()
        disagreements.append(1.0 - dice)
    return float(np.mean(disagreements))


def init_acms_state():
    return {
        'ema_adj_disagreement': None,
        'below_threshold_epochs': 0,
        'switched': False,
        'switch_epoch': None,
        'last_mean_adj_disagreement': None,
    }


def get_active_mutation_terms(opt):
    if not getattr(opt, 'enable_acms', False):
        return ['p1', 'p2', 'p3', 'p4', 'p1234'], 'baseline'
    if opt._acms_state['switched']:
        return ['p1', 'p2', 'p3', 'p4', 'p1234'], 'pool_b'
    return ['p1', 'p2', 'p3', 'p4'], 'pool_a'


def update_acms_state(opt, epoch, epoch_mean_adj_disagreement):
    state = opt._acms_state
    state['last_mean_adj_disagreement'] = epoch_mean_adj_disagreement
    if state['ema_adj_disagreement'] is None:
        state['ema_adj_disagreement'] = epoch_mean_adj_disagreement
    else:
        momentum = opt.acms_ema_momentum
        state['ema_adj_disagreement'] = (
            momentum * state['ema_adj_disagreement']
            + (1.0 - momentum) * epoch_mean_adj_disagreement
        )

    switched_this_epoch = False
    if not state['switched'] and epoch >= opt.acms_min_epoch:
        if state['ema_adj_disagreement'] <= opt.acms_threshold:
            state['below_threshold_epochs'] += 1
        else:
            state['below_threshold_epochs'] = 0

        if state['below_threshold_epochs'] >= opt.acms_patience:
            state['switched'] = True
            state['switch_epoch'] = epoch
            switched_this_epoch = True

    return switched_this_epoch

def test(model, path, dataset, opt):
    data_path = os.path.join(path, dataset)
    image_root = f'{data_path}/images/'
    gt_root = f'{data_path}/masks/'
    model.eval()
    
    test_loader = get_loader(
        image_root=image_root, gt_root=gt_root, 
        batchsize=opt.test_batchsize, trainsize=opt.img_size,
        shuffle=False, num_workers=opt.test_num_workers, split='test', color_image=opt.color_image
    )
    
    DSC, IOU, total_images = 0.0, 0.0, 0
    with torch.no_grad():
        for pack in test_loader:
            images, gts, original_shapes, _ = pack       
            images = images.cuda()
            gts = gts.cuda().float()

            ress = model(images)
            if not isinstance(ress, list):
                ress = [ress]
            # Take the primary output
            predictions = ress[-1]
            
            for i in range(len(images)):
                # Note: original_shapes in some loaders is [W, H], in others [H, W]
                # We ensure it matches your specific data loader's return order
                h_orig, w_orig = int(original_shapes[0][i]), int(original_shapes[1][i])
                
                # 1. Prediction Resize (Bilinear for soft maps)
                p = predictions[i].unsqueeze(0)
                pred_resized = F.interpolate(p, size=(h_orig, w_orig), mode='bilinear', align_corners=False)
                pred_resized = pred_resized.sigmoid().squeeze()
                
                # 2. Local Normalization
                pred_resized = (pred_resized - pred_resized.min()) / (pred_resized.max() - pred_resized.min() + 1e-8)
                
                # 3. GT Resize (NEAREST to maintain binary mask integrity)
                g = gts[i].unsqueeze(0)
                gt_resized = F.interpolate(g, size=(h_orig, w_orig), mode='nearest').squeeze()

                #print(pred_resized.shape, gt_resized.shape, g.shape)

                # 4. Binary Thresholding
                input_binary = (pred_resized >= 0.5).float()
                target_binary = (gt_resized >= 0.2).float() 

                # Applying original thresholding (0.5 for pred, 0.2 for target) 
                DSC += dice_coefficient(input_binary, target_binary).item()
                IOU += iou(input_binary, target_binary).item()
                total_images += 1

    return DSC / total_images, IOU / total_images, total_images

def train(train_loader, model, optimizer, epoch, opt, model_name):
    model.train()
    global best, test_dice_at_best_val, total_train_time, dict_plot
    
    epoch_start = time.time()
    loss_record = AvgMeter()
    size_rates = [0.75, 1, 1.25] 
    total_step = len(train_loader)
    epoch_adj_disagreements = []

    for i, (images, gts) in enumerate(train_loader, start=1):
        for rate in size_rates:            
            optimizer.zero_grad()
            images, gts = Variable(images).cuda(), Variable(gts).float().cuda()
    
            if rate != 1:
                trainsize = int(round(opt.img_size * rate / 32) * 32)
                images = F.interpolate(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.interpolate(gts, size=(trainsize, trainsize), mode='nearest')
            
            P = model(images)
            if not isinstance(P, list):
                P = [P]
            loss_p1 = structure_loss(P[0], gts)
            loss_p2 = structure_loss(P[1], gts)
            loss_p3 = structure_loss(P[2], gts)
            loss_p4 = structure_loss(P[3], gts)
            loss_p1234 = structure_loss(P[0]+P[1]+P[2]+P[3], gts)
            loss_terms = {
                'p1': loss_p1,
                'p2': loss_p2,
                'p3': loss_p3,
                'p4': loss_p4,
                'p1234': loss_p1234,
            }
            active_terms, acms_pool = get_active_mutation_terms(opt)
            loss_scale = len(loss_terms) / float(len(active_terms))
            loss = loss_scale * sum(loss_terms[name] for name in active_terms)

            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            
            if rate == 1:
                loss_record.update(loss.data, opt.batchsize)
                if opt.enable_acms:
                    epoch_adj_disagreements.append(adjacent_head_disagreement(P))
                
        if i % 100 == 0 or i == total_step:
            print(f'{datetime.now()} Epoch [{epoch:03d}/{opt.epoch:03d}], Step [{i:04d}/{total_step:04d}], '
                  f'LR: {optimizer.param_groups[0]["lr"]:.6f}, Loss: {loss_record.show():.4f}')
        
    total_train_time += (time.time() - epoch_start)
    if opt.enable_acms:
        epoch_mean_adj_disagreement = float(np.mean(epoch_adj_disagreements)) if epoch_adj_disagreements else 0.0
        switched_this_epoch = update_acms_state(opt, epoch, epoch_mean_adj_disagreement)
        logging.info(
            "ACMS epoch %d | pool=%s | mean_adj_disagreement=%.6f | ema_adj_disagreement=%.6f | below_threshold_epochs=%d | switched=%s",
            epoch,
            'pool_b' if opt._acms_state['switched'] else 'pool_a',
            epoch_mean_adj_disagreement,
            opt._acms_state['ema_adj_disagreement'],
            opt._acms_state['below_threshold_epochs'],
            switched_this_epoch,
        )
        print(
            f"ACMS epoch {epoch:03d} | pool={'pool_b' if opt._acms_state['switched'] else 'pool_a'} | "
            f"mean_adj_disagreement={epoch_mean_adj_disagreement:.6f} | "
            f"ema_adj_disagreement={opt._acms_state['ema_adj_disagreement']:.6f}"
        )
    
    # Save Last
    save_path = opt.train_save
    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_path, f"{model_name}-last.pth"))

    # Validation and Testing
    epoch_results = {}
    for ds in ['test', 'val']:
        d_dice, d_iou, _ = test(model, opt.test_path, ds, opt)
        epoch_results[ds] = d_dice
        logging.info(f'Epoch: {epoch}, Dataset: {ds}, Dice: {d_dice:.4f}, IoU: {d_iou:.4f}')
        print(f'Epoch: {epoch}, Dataset: {ds}, Dice: {d_dice:.4f}, IoU: {d_iou:.4f}')
        dict_plot[ds].append(d_dice)

    # Check if Best Validation Dice
    if epoch_results['val'] > best:
        logging.info(f"### Best Model Saved (Dice improved from {best:.4f} to {epoch_results['val']:.4f}) ###")
        print(f"### Best Model Saved (Dice improved from {best:.4f} to {epoch_results['val']:.4f}) ###")
        best = epoch_results['val']
        test_dice_at_best_val = epoch_results['test'] # Track test dice at peak val
        torch.save(model.state_dict(), os.path.join(save_path, f"{model_name}-best.pth"))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # network related parameters
    parser.add_argument('--encoder', type=str,
                        default='pvt_v2_b2', help='Name of encoder: pvt_v2_b2, pvt_v2_b0, resnet18, resnet34 ...')
    parser.add_argument('--expansion_factor', type=int,
                        default=2, help='expansion factor in MSCB block')
    parser.add_argument('--kernel_sizes', type=int, nargs='+',
                        default=[1, 3, 5], help='multi-scale kernel sizes in MSDC block')
    parser.add_argument('--lgag_ks', type=int,
                        default=3, help='Kernel size in LGAG')
    parser.add_argument('--activation_mscb', type=str,
                        default='relu6', help='activation used in MSCB: relu6 or relu')
    parser.add_argument('--no_dw_parallel', action='store_true', 
                        default=False, help='use this flag to disable depth-wise parallel convolutions')
    parser.add_argument('--concatenation', action='store_true', 
                        default=False, help='use this flag to concatenate feature maps in MSDC block')
    parser.add_argument('--no_pretrain', action='store_true', 
                        default=False, help='use this flag to turn off loading pretrained enocder weights')
    parser.add_argument('--pretrained_dir', type=str,
                        default='./pretrained_pth/pvt/', help='path to pretrained encoder dir')
    parser.add_argument('--supervision', type=str,
                    default='mutation', help='loss supervision: mutation, deep_supervision or last_layer')    
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.0005) # base learning rate is 0.0005 for CosineAnnealingLR and 0.0001 for no scheduler
    parser.add_argument('--batchsize', type=int, default=8)
    parser.add_argument('--test_batchsize', type=int, default=8)
    parser.add_argument('--img_size', type=int, default=352)
    parser.add_argument('--clip', type=float, default=0.5)
    parser.add_argument('--decay_rate', type=float, default=0.1)
    parser.add_argument('--decay_epoch', type=int, default=300)
    parser.add_argument('--dataset_name', type=str, default='ClinicDB')
    parser.add_argument('--num_runs', type=int, default=5)
    parser.add_argument('--start_run', type=int, default=1)
    parser.add_argument('--seed', type=int, default=2222)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--test_num_workers', type=int, default=4)
    parser.add_argument('--color_image', default=True)
    parser.add_argument('--augmentation', default=True)
    parser.add_argument('--train_path', type=str, default=None)
    parser.add_argument('--test_path', type=str, default=None)
    parser.add_argument('--train_save', type=str, default='') 
    parser.add_argument('--enable_acms', action='store_true', default=False)
    parser.add_argument('--acms_threshold', type=float, default=0.12)
    parser.add_argument('--acms_patience', type=int, default=3)
    parser.add_argument('--acms_min_epoch', type=int, default=20)
    parser.add_argument('--acms_ema_momentum', type=float, default=0.9)
    opt = parser.parse_args()

    if opt.train_path is None:
        opt.train_path = f'./data/polyp/target/{opt.dataset_name}/train'
    if opt.test_path is None:
        opt.test_path = f'./data/polyp/target/{opt.dataset_name}'
    if opt.enable_acms and opt.supervision != 'mutation':
        raise ValueError('ACMS currently supports only supervision=mutation.')

    for run in range(opt.start_run, opt.start_run + opt.num_runs):
        dict_plot = {'val': [], 'test': []}
        best = 0.0
        test_dice_at_best_val = 0.0
        total_train_time = 0
        run_seed = opt.seed + (run - opt.start_run)
        set_seed(run_seed)
        opt._acms_state = init_acms_state()

        if opt.concatenation:
            aggregation = 'concat'
        else: 
            aggregation = 'add'
        
        if opt.no_dw_parallel:
            dw_mode = 'series'
        else: 
            dw_mode = 'parallel'

        timestamp = time.strftime('%H%M%S')
        run_id = (f"{opt.dataset_name}_{opt.encoder}_EMCAD_kernel_sizes_{opt.kernel_sizes}_dw_{dw_mode}_{aggregation}_lgag_ks_{opt.lgag_ks}_ef{opt.expansion_factor}_act_mscb_{opt.activation_mscb}_bs{opt.batchsize}_cas_lr{opt.lr}_"
                      f"e{opt.epoch}_aug{opt.augmentation}_run{run}_seed{run_seed}")
        if opt.enable_acms:
            run_id += f"_acms_thr{str(opt.acms_threshold).replace('.', 'p')}_pat{opt.acms_patience}_min{opt.acms_min_epoch}"
        run_id += f"_t{timestamp}"
        run_id = run_id.replace('[', '').replace(']', '').replace(', ', '_')
        opt.train_save = f'./model_pth/{run_id}/'
        
        os.makedirs('logs', exist_ok=True)
        os.makedirs(opt.train_save, exist_ok=True)
        
        logging.basicConfig(filename=f'logs/train_log_{run_id}.log', level=logging.INFO, 
                            format='[%(asctime)s] %(message)s', force=True)
        logging.info('Run seed: %d', run_seed)
        logging.info('ACMS enabled: %s', opt.enable_acms)


        # Build model
        #model = EMCADNet(dw_parallel=dw_parallel, expansion_factor=expansion_factor, add=add, kernel_sizes=kernel_sizes, att_ks=att_ks, activation=activation, encoder=encoder, pretrain=pretrain, head=head, bbox=False, cds=False) # head='SAH'
        model = EMCADNet(num_classes=1, kernel_sizes=opt.kernel_sizes, expansion_factor=opt.expansion_factor, dw_parallel=not opt.no_dw_parallel, add=not opt.concatenation, lgag_ks=opt.lgag_ks, activation=opt.activation_mscb, encoder=opt.encoder, pretrain= not opt.no_pretrain, pretrained_dir=opt.pretrained_dir)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        '''if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)'''

        model.to(device)

        print(f"Encoder: {opt.encoder} | Decoder: EMCAD")
        cal_params_flops(model, opt.img_size, logging)
        optimizer = torch.optim.AdamW(model.parameters(), opt.lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=opt.epoch, eta_min=1e-6)

        train_loader = get_loader(
            image_root=f'{opt.train_path}/images/', gt_root=f'{opt.train_path}/masks/',
            batchsize=opt.batchsize, trainsize=opt.img_size, 
            shuffle=True, num_workers=opt.num_workers, augmentation=opt.augmentation, split='train', color_image=opt.color_image
        )

        for epoch in range(1, opt.epoch + 1):
            adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
            train(train_loader, model, optimizer, epoch, opt, run_id)
            scheduler.step()
        # FINAL SUMMARY
        
        summary = (f"\n{'='*40}\nFINAL RESULTS: {run_id}\n"
                   f"Best Val Dice: {best:.4f}\n"
                   f"Test Dice at Best Val: {test_dice_at_best_val:.4f}\n"
                   f"Total Train Time: {total_train_time:.2f}s\n{'='*40}")
        print(summary)
        logging.info(summary)
        
        
