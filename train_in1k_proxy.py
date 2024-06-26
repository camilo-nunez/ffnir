import argparse
import os
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

from config.init import create_train_in1k_config
from model.lgffem import LGFFEM

import torch
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from torchvision.transforms import v2
from torchvision.datasets import ImageNet

from pytorch_metric_learning import losses

import numpy as np

## Customs configs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

## Restriction
AVAILABLE_SIZES = [96, 224]

def parse_option():
    parser = argparse.ArgumentParser(
        'Thesis cnunezf training LGFFEM model over ImageNet-1k using SubCenterArcFaceLoss.', add_help=True)
    
    parser.add_argument('--cfg_model_backbone',
                        type=str,
                        required=True,
                        metavar="FILE",
                        help='Path to BACKBONE config file. Must be a YAML file.'
                       )
    parser.add_argument('--cfg_model_neck',
                        type=str,
                        required=True,
                        metavar="FILE",
                        help='Path to NECK config file. Must be a YAML file.'
                       )
    parser.add_argument('--cfg_model_head',
                        type=str,
                        required=True,
                        metavar="FILE",
                        help='Path to HEAD config file. Must be a YAML file.'
                       )
    
    parser.add_argument('--summary',
                        action='store_true',
                        help="Display the summary of the model."
                       )
    
    parser.add_argument('--num_epochs',
                        type=int,
                        default=25,
                        help='Default 25 epochs.'
                       )
    parser.add_argument('--batch_size',
                        type=int,
                        default=512,
                        help='Default 512 epochs.'
                       )
    
    parser.add_argument('--dataset_path',
                        type=str,
                        default='/thesis/classical/imagenet-1k', 
                        help='Path to complete DATASET.'
                       )
    
    parser.add_argument('--lr', 
                        type=float, 
                        default=1e-2,
                        help='Learning rate used by the \'adamw\' optimizer. Default is 1e-2.'
                       )
    parser.add_argument('--wd', 
                        type=float, 
                        default=1e-5,
                        help='Weight decay used by the \'adamw\' optimizer. Default is 1e-5.'
                       )
    parser.add_argument('--optimizer', 
                        type=str, 
                        default='adamw',
                        help='The optimizer to use. The available opts are: \'adamw\' or \'sgd\'. By default its \'adamw\'.'
                       )
    
    parser.add_argument('--amp',
                        action='store_true',
                        help="Enable Automatic Mixed Precision train."
                       )
    
    parser.add_argument('--scheduler',
                        action='store_true',
                        help="Use scheduler."
                       )
    parser.add_argument('--scheduler_eta_min', 
                        type=float, 
                        default=1e-4,
                        help='Minimum learning rate. Default is 1e-4.',
                       )

    parser.add_argument('--checkpoint',
                        type=str,
                        metavar="FILE",
                        default = None,
                        help="Checkpoint filename.",
                       )
    parser.add_argument('--checkpoint_path',
                        type=str,
                        default='/thesis/checkpoint', 
                        help='Path to complete DATASET.',
                       )
    
    parser.add_argument('--new_train',
                        action='store_true',
                       )
    parser.add_argument('--new_lr',
                        action='store_true',
                       )
    
    parser.add_argument('--loss_m', 
                        type=float, 
                        default=17.5,
                        help='The angular margin penalty in degrees or factor \'m\' in the loss equation. Default is 17.5 degress (0.3 rad).',
                       )
    parser.add_argument('--loss_s', 
                        type=int, 
                        default=64,
                        help='Scale factor or factor \'s\' in the loss equation. Default is 64.'
                       )
    parser.add_argument('--loss_sc', 
                        type=int, 
                        default=6,
                        help='The number of sub centers per class. Default is 6.',
                       )
    
    parser.add_argument('--img_shape', 
                        type=int, 
                        default=96,
                        help=f'Image size for the transform \'RandomResizedCrop\'. The available size are: {AVAILABLE_SIZES}. Default is 96.',
                       )
    
    args, unparsed = parser.parse_known_args()
    config = create_train_in1k_config(args)

    return args, config

if __name__ == '__main__':
    
    # Load configs for the model and the dataset
    args, base_config = parse_option()
    
    # Writer will output to ./runs/ directory by default
    writer = SummaryWriter()
    
    # Check the principal exceptions
    if not torch.cuda.is_available(): raise Exception('This script is only available to run in GPU.')
    if args.img_shape not in AVAILABLE_SIZES: raise Exception(f'The image size choose for the resize transform is not available. The available size are: {AVAILABLE_SIZES}')
    if args.loss_m>20.1: raise Exception(f'The angular margin penalty must be less than 0.35rad (20.1 degrees) to avoid saturated values. Please refer to the main paper or CosFaceLoss paper.')

    # Create the backbone and neck model
    print(f'[+] Configuring base model with variables: {base_config.MODEL}')
    base_model = LGFFEM(base_config).to(device)
    print('[+] Ready !')
    
    # Display the summary of the net
    if args.summary: summary(base_model)
        
    # Load the dataset
    print(f'[+] Loading ImageNet-1k ataset...')
    print(f'[++] Using batch_size: {args.batch_size}')
    
    train_transforms = v2.Compose([
                                    v2.ToImage(),
                                    v2.RandomResizedCrop(size=(args.img_shape, args.img_shape), antialias=True),
                                    v2.RandomHorizontalFlip(p=0.5),
                                    v2.ToDtype(torch.float32, scale=True),
                                    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                  ])
    
    train_dataset = ImageNet(root=args.dataset_path, split='train', transform = train_transforms)

    training_params = {'batch_size': args.batch_size,
                       'shuffle': True,
                       'drop_last': True,
                       'collate_fn': lambda batch: tuple(zip(*batch)),
                       'num_workers': 4,
                       'pin_memory':True,
                      }

    training_loader = torch.utils.data.DataLoader(train_dataset, **training_params)
    print('[+] Ready !')
    
    # === General train variables ===
    print('[+] Preparing training configuration...')
    ## Config the optimizer
    params = [p for p in base_model.parameters() if p.requires_grad]
    
    if args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(params,
                                      lr=args.lr,
                                      weight_decay=args.wd)
        print(f'[++] Using AdamW optimizer. Configs: lr->{args.lr}, weight_decay->{args.wd}')
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.wd)
        print(f'[++] Using SGD optimizer.')
    else:
        raise Exception("The optimizer selected doesn't exist. The available optis are: \'adamw\' or \'sgd\'.") 
        
    start_epoch = 1
    end_epoch = args.num_epochs
    best_loss = 1e5
    global_steps = 0
    
    ## Prepare Automatic Mixed Precision
    if args.amp:
        print("[++] Using Automatic Mixed Precision")
        use_amp = True
    else:
        use_amp = False
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    ## Prepare Loss Metric
#     loss_func = losses.SubCenterArcFaceLoss(num_classes=1000, embedding_size=4*base_config.MODEL.NECK.NUM_CHANNELS, margin=args.loss_m, scale=args.loss_m, sub_centers=args.loss_sc).to(device)
    loss_func = losses.ProxyAnchorLoss(num_classes=1000, embedding_size=4*base_config.MODEL.NECK.NUM_CHANNELS, margin = 0.1, alpha = 32).to(device)
#     loss_optimizer = torch.optim.AdamW(loss_func.parameters(), lr=1e-4)
    loss_optimizer = torch.optim.SGD(loss_func.parameters(), lr=0.01)
    print(f"[++] Using ProxyAnchorLoss. embedding_size->{4*base_config.MODEL.NECK.NUM_CHANNELS}, margin->{args.loss_m}, scale->{args.loss_s}, sub_centers->{args.loss_sc}")
    ### Display the summary of the loss_func net
    if args.summary: summary(loss_func)  
    
    ## Load the checkpoint if is need it
    if args.checkpoint:
        print('[++] Loading checkpoint...')
        checkpoint = torch.load(os.path.join(args.checkpoint))
        
        match_n = base_model.neck.load_state_dict(checkpoint['model_neck_state_dict'], strict = False)
        print('[++] Loaded neck weights.',match_n)
        match_h = base_model.head.load_state_dict(checkpoint['model_head_state_dict'], strict = False)
        print('[++] Loaded head weights.',match_h)
        
        if not args.new_train:
            match_loss = loss_func.load_state_dict(checkpoint['loss_state_dict'], strict = False)
            print('[++] Loaded loss_func weights.',match_loss)
            
            if not args.new_lr:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print('[++] Loaded optimizer.')
                loss_optimizer.load_state_dict(checkpoint['loss_optimizer_state_dict'])
                print('[++] Loaded loss_optimizer optimizer.')

            best_loss = checkpoint['best_loss']
            start_epoch = checkpoint['epoch'] + 1

        print(f'[++] Ready. start_epoch: {start_epoch} - best_loss: {best_loss}')
        
    ## Scheduler
    if args.scheduler:
        print(f"[+] Using \'CosineAnnealingWarmRestarts\'. eta_min->{args.scheduler_eta_min}")
        if args.checkpoint and not (args.new_lr or args.new_train):
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=1, eta_min=args.scheduler_eta_min, last_epoch=start_epoch)
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print('[++] Loaded scheduler.')
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=1, eta_min=args.scheduler_eta_min)

    print('[+] Ready !')
    
    # === Train the model ===
    print('[+] Starting training ...')
    start_t = datetime.now()
    
    base_model.train()
    
    for e, epoch in enumerate(range(start_epoch, end_epoch + 1)):
        loss_l = []
        with tqdm(training_loader, unit=" batch") as tepoch:
            for batch_idx, (data, labels) in enumerate(tepoch):
                
                optimizer.zero_grad(set_to_none=True)
                loss_optimizer.zero_grad(set_to_none=True)

                data = torch.stack(data).to(device)
                labels = torch.tensor(labels, dtype=torch.int64).to(device)

                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                    embeddings = base_model(data)
                    loss = loss_func(embeddings, labels)
                    
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(base_model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.step(loss_optimizer)
                scaler.update()
                
                if args.scheduler:
                    current_lr = scheduler.get_last_lr()[0]
                else:
                    current_lr = optimizer.param_groups[0]['lr']
                
                loss_l.append(loss.detach().cpu())
                loss_mean = np.mean(np.array(loss_l))
                
                description_s = 'Epoch: {}/{}. lr: {:1.10f} loss_mean: {:1.10f}'\
                                   .format(epoch, end_epoch, current_lr, loss_mean)
                
                tepoch.set_description(description_s)
                
                ## to board
                writer.add_scalar('learning_rate', current_lr, global_steps)
                writer.add_scalar('loss_mean', loss_mean, global_steps)
                
                global_steps+=1

                if args.scheduler:
                    scheduler.step(e + batch_idx/len(tepoch))

        if loss_mean < best_loss:
            best_loss = loss_mean

            torch.save({'model_neck_state_dict': base_model.neck.state_dict(),
                        'model_head_state_dict': base_model.head.state_dict(),
                        'loss_state_dict': loss_func.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss_optimizer_state_dict': loss_optimizer.state_dict(),
                        'scheduler_state_dict':scheduler.state_dict() if args.scheduler else None,
                        'epoch': epoch,
                        'best_loss': best_loss,
                        'fn_cfg_model_head': str(args.cfg_model_head), 
                        'fn_cfg_model_backbone': str(args.cfg_model_backbone),
                        'fn_cfg_model_neck': str(args.cfg_model_neck),
                       },
                       os.path.join(args.checkpoint_path, f'{datetime.utcnow().strftime("%Y%m%d_%H%M")}-EMB-{Path(args.cfg_model_backbone).stem}-{Path(args.cfg_model_neck).stem}-{Path(args.cfg_model_head).stem}-epoch{epoch}.pth'))
                    
    end_t = datetime.now()
    print('[+] Ready, the train phase took:', (end_t - start_t))
    
    writer.close()
