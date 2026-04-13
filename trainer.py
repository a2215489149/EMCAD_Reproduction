import argparse
import csv
import json
import logging
import os
import random
import sys
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms

from utils.dataset_synapse import Synapse_dataset, RandomGenerator
from utils.utils import powerset, one_hot_encoder, DiceLoss, val_single_volume, test_single_volume


EXPERIMENT_FIELDNAMES = [
    "task",
    "dataset",
    "model",
    "run",
    "paper_metric",
    "reproduced_metric",
    "improved_metric",
    "delta",
    "hardware",
    "elapsed",
    "command",
    "checkpoint",
    "notes",
]


def _run_official_synapse_test(args, model):
    db_test = Synapse_dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir, nclass=args.num_classes)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=args.test_num_workers)
    logging.info("Target-stop full test: %d test volumes", len(testloader))
    model.eval()
    metric_list = 0.0
    for _, sampled_batch in tqdm(enumerate(testloader), total=len(testloader), ncols=70):
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        metric_i = test_single_volume(
            image,
            label,
            model,
            classes=args.num_classes,
            patch_size=[args.img_size, args.img_size],
            test_save_path=None,
            case=case_name,
            z_spacing=args.z_spacing,
        )
        metric_list += np.array(metric_i)

    metric_list = metric_list / len(db_test)
    mean_metrics = np.mean(metric_list, axis=0)
    return {
        "mean_dice": float(mean_metrics[0]),
        "mean_hd95": float(mean_metrics[1]),
        "mean_jacard": float(mean_metrics[2]),
        "mean_asd": float(mean_metrics[3]),
    }


def _append_target_stop_record(args, snapshot_path, checkpoint_path, epoch_num, val_dice, test_metrics, elapsed_seconds):
    csv_path = Path(args.target_stop_record_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    paper_dice = float(args.paper_target_dice)
    test_dice = float(test_metrics["mean_dice"] * 100.0)
    delta = test_dice - paper_dice
    hardware = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    with csv_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=EXPERIMENT_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "task": "target_stop",
                "dataset": args.dataset,
                "model": f"{args.encoder}_emcad",
                "run": f"epoch_{epoch_num + 1}",
                "paper_metric": f"{paper_dice:.2f}",
                "reproduced_metric": f"{test_dice:.4f}",
                "improved_metric": "",
                "delta": f"{delta:.4f}",
                "hardware": hardware,
                "elapsed": f"{elapsed_seconds:.2f}s",
                "command": getattr(args, "command_line", ""),
                "checkpoint": checkpoint_path,
                "notes": (
                    f"target-stop accepted; quick_val_dice={val_dice * 100.0:.4f}; "
                    f"mean_hd95={test_metrics['mean_hd95']:.4f}; "
                    f"mean_jacard={test_metrics['mean_jacard'] * 100.0:.4f}; "
                    f"mean_asd={test_metrics['mean_asd']:.4f}"
                ),
            }
        )

    details_path = Path(snapshot_path) / "target_stop_result.json"
    details_path.write_text(
        json.dumps(
            {
                "epoch": epoch_num + 1,
                "paper_target_dice": paper_dice,
                "quick_validation_dice": round(val_dice * 100.0, 6),
                "official_test": {
                    "mean_dice": round(test_dice, 6),
                    "mean_hd95": round(test_metrics["mean_hd95"], 6),
                    "mean_jacard": round(test_metrics["mean_jacard"] * 100.0, 6),
                    "mean_asd": round(test_metrics["mean_asd"], 6),
                },
                "delta_vs_paper": round(delta, 6),
                "checkpoint": checkpoint_path,
                "command": getattr(args, "command_line", ""),
                "record_csv": str(csv_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _should_run_target_stop_test(args, epoch_num, val_dice, best_probe_dice):
    if not getattr(args, "enable_target_stop", False):
        return False
    if epoch_num + 1 < args.target_stop_min_epoch:
        return False
    trigger_threshold = (args.paper_target_dice - args.val_trigger_margin) / 100.0
    if val_dice < trigger_threshold:
        return False
    probe_delta = args.target_stop_retest_delta / 100.0
    return val_dice >= (best_probe_dice + probe_delta)


def _model_state_dict(model):
    return model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()


def _load_model_state_dict(model, state_dict):
    try:
        model.load_state_dict(state_dict)
        return
    except RuntimeError:
        pass

    if state_dict and all(key.startswith("module.") for key in state_dict.keys()):
        stripped_state = {key[len("module."):]: value for key, value in state_dict.items()}
        model.load_state_dict(stripped_state)
        return

    if isinstance(model, nn.DataParallel):
        prefixed_state = {f"module.{key}": value for key, value in state_dict.items()}
        model.load_state_dict(prefixed_state)
        return

    model.load_state_dict(state_dict)


def _move_optimizer_state_to_device(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _save_resume_state(
    resume_path,
    model,
    optimizer,
    epoch_num,
    iter_num,
    best_performance,
    best_target_stop_probe,
    elapsed_seconds,
):
    torch.save(
        {
            "model_state_dict": _model_state_dict(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": int(epoch_num),
            "next_epoch": int(epoch_num + 1),
            "iter_num": int(iter_num),
            "best_performance": float(best_performance),
            "best_target_stop_probe": float(best_target_stop_probe),
            "elapsed_seconds": float(elapsed_seconds),
        },
        resume_path,
    )


def _load_resume_state(args, model, optimizer, device):
    if not getattr(args, "resume_checkpoint", None):
        return 0, 0, 0.0, -1.0, 0.0

    resume_path = Path(args.resume_checkpoint)
    if not resume_path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

    checkpoint = torch.load(resume_path, map_location=device)
    start_epoch = int(args.resume_epoch) if args.resume_epoch is not None else 0
    iter_num = int(args.resume_iter) if args.resume_iter is not None else 0
    best_performance = (
        float(args.resume_best_performance)
        if args.resume_best_performance is not None
        else 0.0
    )
    best_target_stop_probe = (
        float(args.resume_best_target_stop_probe)
        if args.resume_best_target_stop_probe is not None
        else -1.0
    )
    elapsed_seconds = 0.0
    resume_mode = "weights_only"

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        _load_model_state_dict(model, checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            _move_optimizer_state_to_device(optimizer, device)
        start_epoch = int(checkpoint.get("next_epoch", checkpoint.get("epoch", -1) + 1))
        iter_num = int(checkpoint.get("iter_num", iter_num))
        best_performance = float(checkpoint.get("best_performance", best_performance))
        best_target_stop_probe = float(
            checkpoint.get("best_target_stop_probe", best_target_stop_probe)
        )
        elapsed_seconds = float(checkpoint.get("elapsed_seconds", 0.0))
        resume_mode = "full_state"
    else:
        _load_model_state_dict(model, checkpoint)

    logging.info(
        "Resumed Synapse training from %s (%s): start_epoch=%d iter_num=%d best_performance=%.6f best_target_stop_probe=%.6f",
        resume_path,
        resume_mode,
        start_epoch,
        iter_num,
        best_performance,
        best_target_stop_probe,
    )
    return start_epoch, iter_num, best_performance, best_target_stop_probe, elapsed_seconds
            
def inference(args, model, best_performance):
    db_test = Synapse_dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir, nclass=args.num_classes)
    
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=args.test_num_workers)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        metric_i = val_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                      case=case_name, z_spacing=args.z_spacing)
        metric_list += np.array(metric_i)
    metric_list = metric_list / len(db_test)
    performance = float(np.mean(metric_list, axis=0))
    logging.info('Testing performance in val model: mean_dice : %f, best_dice : %f' % (performance, best_performance))
    return performance

def trainer_synapse(args, model, snapshot_path):
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    grad_accum_steps = max(1, args.grad_accum_steps)
    
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train", nclass=args.num_classes,
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    worker_fn = worker_init_fn if args.num_workers > 0 else None
    trainloader = DataLoader(
        db_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=not args.no_pin_memory,
        worker_init_fn=worker_fn,
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1 and args.n_gpu > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.to(device)

    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)

    #optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    logging.info(
        "Physical batch size: %d | Grad accumulation: %d | Effective batch size: %d",
        batch_size,
        grad_accum_steps,
        batch_size * grad_accum_steps,
    )
    start_epoch, iter_num, best_performance, best_target_stop_probe, elapsed_offset = _load_resume_state(
        args,
        model,
        optimizer,
        device,
    )
    target_stop_accept_threshold = (args.paper_target_dice - args.test_accept_margin) / 100.0
    train_start_time = time.time() - elapsed_offset
    iterator = tqdm(range(start_epoch, max_epoch), ncols=70, total=max_epoch, initial=start_epoch)
    ss = None
    
    for epoch_num in iterator:
        optimizer.zero_grad()
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.squeeze(1).cuda()
            
            P = model(image_batch, mode='train')

            if  not isinstance(P, list):
                P = [P]
            if ss is None:
                n_outs = len(P)
                out_idxs = list(np.arange(n_outs)) #[0, 1, 2, 3]#, 4, 5, 6, 7]
                if args.supervision == 'mutation':
                    ss = [x for x in powerset(out_idxs)]
                elif args.supervision == 'deep_supervision':
                    ss = [[x] for x in out_idxs]
                else:
                    ss = [[-1]]
                print(ss)
            
            loss = 0.0
            w_ce, w_dice = 0.3, 0.7
          
            for s in ss:
                iout = 0.0
                if(s==[]):
                    continue
                for idx in range(len(s)):
                    iout += P[s[idx]]
                loss_ce = ce_loss(iout, label_batch[:].long())
                loss_dice = dice_loss(iout, label_batch, softmax=True)
                loss += (w_ce * loss_ce + w_dice * loss_dice)
            
            loss_value = loss.item()
            (loss / grad_accum_steps).backward()
            if ((i_batch + 1) % grad_accum_steps == 0) or (i_batch + 1 == len(trainloader)):
                optimizer.step()
                optimizer.zero_grad()
            #lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9 # we did not use this
            lr_ = base_lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss_value, iter_num)
            

            if iter_num % 50 == 0:
                logging.info('iteration %d, epoch %d : loss : %f, lr: %f' % (iter_num, epoch_num, loss_value, lr_))
                
        logging.info('iteration %d, epoch %d : loss : %f, lr: %f' % (iter_num, epoch_num, loss_value, lr_))
        
        save_mode_path = os.path.join(snapshot_path, 'last.pth')
        torch.save(_model_state_dict(model), save_mode_path)
        _save_resume_state(
            os.path.join(snapshot_path, 'last_resume.pt'),
            model,
            optimizer,
            epoch_num,
            iter_num,
            best_performance,
            best_target_stop_probe,
            time.time() - train_start_time,
        )
        
        performance = inference(args, model, best_performance)
        
        save_interval = args.save_interval

        if(best_performance <= performance):
            best_performance = performance
            save_mode_path = os.path.join(snapshot_path, 'best.pth')
            torch.save(_model_state_dict(model), save_mode_path)
            _save_resume_state(
                os.path.join(snapshot_path, 'best_resume.pt'),
                model,
                optimizer,
                epoch_num,
                iter_num,
                best_performance,
                best_target_stop_probe,
                time.time() - train_start_time,
            )
            logging.info("save model to {}".format(save_mode_path))
            
        if (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(_model_state_dict(model), save_mode_path)
            _save_resume_state(
                os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '_resume.pt'),
                model,
                optimizer,
                epoch_num,
                iter_num,
                best_performance,
                best_target_stop_probe,
                time.time() - train_start_time,
            )
            logging.info("save model to {}".format(save_mode_path))

        if _should_run_target_stop_test(args, epoch_num, performance, best_target_stop_probe):
            best_target_stop_probe = performance
            logging.info(
                "Target-stop probe triggered at epoch %d: quick val Dice %.4f vs paper %.4f. Running official Synapse test.",
                epoch_num + 1,
                performance * 100.0,
                args.paper_target_dice,
            )
            official_test_metrics = _run_official_synapse_test(args, model)
            logging.info(
                "Target-stop full test result: mean_dice %.4f mean_hd95 %.4f mean_jacard %.4f mean_asd %.4f",
                official_test_metrics["mean_dice"] * 100.0,
                official_test_metrics["mean_hd95"],
                official_test_metrics["mean_jacard"] * 100.0,
                official_test_metrics["mean_asd"],
            )
            if official_test_metrics["mean_dice"] >= target_stop_accept_threshold:
                target_stop_path = os.path.join(snapshot_path, 'target_stop.pth')
                torch.save(_model_state_dict(model), target_stop_path)
                _save_resume_state(
                    os.path.join(snapshot_path, 'target_stop_resume.pt'),
                    model,
                    optimizer,
                    epoch_num,
                    iter_num,
                    best_performance,
                    best_target_stop_probe,
                    time.time() - train_start_time,
                )
                _append_target_stop_record(
                    args=args,
                    snapshot_path=snapshot_path,
                    checkpoint_path=target_stop_path,
                    epoch_num=epoch_num,
                    val_dice=performance,
                    test_metrics=official_test_metrics,
                    elapsed_seconds=time.time() - train_start_time,
                )
                logging.info(
                    "Target-stop accepted at epoch %d. Saved %s and recorded the result to %s.",
                    epoch_num + 1,
                    target_stop_path,
                    args.target_stop_record_csv,
                )
                iterator.close()
                break
            logging.info(
                "Target-stop full test did not meet acceptance threshold %.4f. Training continues.",
                target_stop_accept_threshold * 100.0,
            )
            model.train()

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(_model_state_dict(model), save_mode_path)
            _save_resume_state(
                os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '_resume.pt'),
                model,
                optimizer,
                epoch_num,
                iter_num,
                best_performance,
                best_target_stop_probe,
                time.time() - train_start_time,
            )
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"
