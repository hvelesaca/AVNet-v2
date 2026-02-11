import os
import argparse
from datetime import datetime
import logging

import torch
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from tensorboardX import SummaryWriter
import torchvision.utils as vutils

from utils.dataloader import get_loader, test_dataset  # Asegúrate que test_dataset devuelve tensores coherentes
from utils.utils import clip_gradient, adjust_lr, AvgMeter
from lib.cod_net import CamouflageDetectionNet
import torch.nn.functional as F

from py_sod_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure  

# =========================
# Utilidades de pérdidas
# =========================
def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()


# =========================
# Visualización (TensorBoard)
# =========================
def denorm(img, mean, std):
    """
    img: BxCxHxW en torch.float
    mean/std: tuplas de longitud C
    Retorna BxCxHxW en [0,1] clamped.
    """
    device = img.device
    mean = torch.tensor(mean, device=device).view(1, -1, 1, 1)
    std = torch.tensor(std, device=device).view(1, -1, 1, 1)
    x = img * std + mean
    return x.clamp(0.0, 1.0)

def log_train_images(writer, step, images, thermals, gts, preds,
                     rgb_stats=((0.485,0.456,0.406),(0.229,0.224,0.225)),
                     th_stats=((0.5,0.5,0.5),(0.5,0.5,0.5))):
    """
    images, thermals: Bx3xHxW (normalizados)
    gts, preds: Bx1xHxW en [0,1]
    """
    rgb_mean, rgb_std = rgb_stats
    th_mean, th_std = th_stats

    with torch.no_grad():
        img_vis = denorm(images, rgb_mean, rgb_std)
        th_vis = denorm(thermals, th_mean, th_std)
        gt_vis = gts
        pr_vis = preds

        grid_img = vutils.make_grid(img_vis, nrow=4)
        grid_th = vutils.make_grid(th_vis, nrow=4)
        grid_gt = vutils.make_grid(gt_vis, nrow=4)
        grid_pr = vutils.make_grid(pr_vis, nrow=4)

        writer.add_image('train/RGB', grid_img, global_step=step)
        writer.add_image('train/Thermal', grid_th, global_step=step)
        writer.add_image('train/GT', grid_gt, global_step=step)
        writer.add_image('train/Pred', grid_pr, global_step=step)

def log_val_images(writer, epoch, image, thermal, gt, pred,
                   rgb_stats=((0.485,0.456,0.406),(0.229,0.224,0.225)),
                   th_stats=((0.5,0.5,0.5),(0.5,0.5,0.5))):
    """
    image, thermal: 1x3xHxW (normalizados)
    gt, pred: 1x1xHxW en [0,1]
    """
    img_vis = denorm(image, *rgb_stats)
    th_vis = denorm(thermal, *th_stats)
    writer.add_image('val/RGB', img_vis[0], global_step=epoch)
    writer.add_image('val/Thermal', th_vis[0], global_step=epoch)
    writer.add_image('val/GT', gt[0], global_step=epoch)
    writer.add_image('val/Pred', pred[0], global_step=epoch)


# =========================
# Validación
# =========================
@torch.no_grad()
def val(model, epoch, save_path, writer, opt, device):
    """
    Evalúa con el mismo preprocesamiento que train. Calcula MAE en torch.
    También guarda mejor modelo y loguea una visualización.
    """
    global best_mae, best_em, best_fm, best_sm, best_wfm, best_epoch

    model.eval()
    mae_sum, em_sum, fm_sum, sm_sum, wfm_sum = 0.0, 0.0, 0.0, 0.0, 0.0
    
    # Inicializar métricas según las banderas
    if opt.use_mae:
        mae_metric = MAE()
    if opt.use_emeasure:
        em_metric = Emeasure()
    if opt.use_fmeasure:
        fm_metric = Fmeasure()
    if opt.use_smeasure:
        sm_metric = Smeasure()
    if opt.use_wfmeasure:
        wfm_metric = WeightedFmeasure()
    

    # test_dataset debe devolver tensores: image:1x3xHxW, thermal:1x3xHxW, gt:1x1xHxW
    test_data = test_dataset(
        image_root=os.path.join(opt.val_path, 'Imgs') + '/',
        thermal_root=os.path.join(opt.val_path, 'Thermal') + '/',
        gt_root=os.path.join(opt.val_path, 'GT') + '/',
        testsize=opt.trainsize
    )

    for i in range(test_data.size):
        image, thermal, gt, name = test_data.load_data()
        # image, thermal: 1x3xH'xW', gt: 1x1xH'xW' (0/1)
        image = image.to(device, non_blocking=True)
        thermal = thermal.to(device, non_blocking=True)
        gt_t = gt.to(device, non_blocking=True)

        # forward
        res_list, res_single = model(image, thermal)  # asumiendo tu modelo devuelve (list, tensor)

        # upsample todas las salidas a la resolución de gt
        target_size = gt_t.shape[-2:]
        
        
        combined = 0.0
        if isinstance(res_list, (list, tuple)) and len(res_list) > 0:
            for o in res_list:
                combined = combined + F.interpolate(o, size=target_size, mode='bilinear', align_corners=False)
        if res_single is not None:
            combined = combined + F.interpolate(res_single, size=target_size, mode='bilinear', align_corners=False)

        pred = torch.sigmoid(combined)  # 1x1xHxW

        
        # Convertir a numpy para py_sod_metrics (HxW, valores en [0, 255])
        pred_np = (pred.squeeze().cpu().numpy() * 255).astype(np.uint8)
        gt_np = (gt_t.squeeze().cpu().numpy() * 255).astype(np.uint8)
        
        # Actualizar métricas según las banderas
        if opt.use_mae:
            mae_metric.step(pred=pred_np, gt=gt_np)
        if opt.use_emeasure:
            em_metric.step(pred=pred_np, gt=gt_np)
        if opt.use_fmeasure:
            fm_metric.step(pred=pred_np, gt=gt_np)
        if opt.use_smeasure:
            sm_metric.step(pred=pred_np, gt=gt_np)
        if opt.use_wfmeasure:
            wfm_metric.step(pred=pred_np, gt=gt_np)
            
        # Log visualización para la primera imagen
        if i == 0:
            log_val_images(writer, epoch, image, thermal, gt_t, pred)    
            
        # Obtener resultados de las métricas
        metrics_results = {}
        print_msg = f'Epoch: {epoch}'
        
        if opt.use_mae:
            mae = mae_metric.get_results()['mae']
            metrics_results['MAE'] = mae
            writer.add_scalar('val/MAE', mae, global_step=epoch)
            print_msg += f', MAE: {mae:.6f}'
        
        if opt.use_emeasure:
            em_results = em_metric.get_results()
            em = em_results['em']['curve'].mean()  # Promedio de la curva E-measure
            metrics_results['Emeasure'] = em
            writer.add_scalar('val/Emeasure', em, global_step=epoch)
            print_msg += f', E-measure: {em:.6f}'
        
        if opt.use_fmeasure:
            fm_results = fm_metric.get_results()
            fm = fm_results['fm']['curve'].mean()  # Promedio de la curva F-measure
            metrics_results['Fmeasure'] = fm
            writer.add_scalar('val/Fmeasure', fm, global_step=epoch)
            print_msg += f', F-measure: {fm:.6f}'
        
        if opt.use_smeasure:
            sm = sm_metric.get_results()['sm']
            metrics_results['Smeasure'] = sm
            writer.add_scalar('val/Smeasure', sm, global_step=epoch)
            print_msg += f', S-measure: {sm:.6f}'
        
        if opt.use_wfmeasure:
            wfm = wfm_metric.get_results()['wfm']
            metrics_results['WeightedFmeasure'] = wfm
            writer.add_scalar('val/WeightedFmeasure', wfm, global_step=epoch)
            print_msg += f', Weighted F-measure: {wfm:.6f}'
        
        # MAE en torch
        mae = torch.mean(torch.abs(pred - gt_t)).item()
        mae_sum += mae

        # Log visualización para la primera imagen
        if i == 0:
            log_val_images(writer, epoch, image, thermal, gt_t, pred)

    mae = mae_sum / max(1, test_data.size)
    writer.add_scalar('val/MAE', mae, global_step=epoch)
    #print(f'Epoch: {epoch}, MAE: {mae:.6f}, bestMAE: {best_mae:.6f}, bestEpoch: {best_epoch}.')
    print_msg += f', bestMAE: {best_mae:.6f}, bestEpoch: {best_epoch}' 
    print(print_msg + '.')

    if epoch == 1:
        best_mae = mae
    else:
        if mae < best_mae:
            best_mae = mae
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(save_path, 'Net_epoch_best.pth'))
            print(f'Save state_dict successfully! Best epoch: {epoch}.')
    #logging.info(f'[Val Info]: Epoch:{epoch} MAE:{mae:.6f} bestEpoch:{best_epoch} bestMAE:{best_mae:.6f}')
    # Logging
    log_msg = f'[Val Info]: Epoch:{epoch}'
    for metric_name, metric_value in metrics_results.items():
        log_msg += f' {metric_name}:{metric_value:.6f}'
    log_msg += f' bestEpoch:{best_epoch} bestMAE:{best_mae:.6f}'
    logging.info(log_msg)
    
# =========================
# Entrenamiento
# =========================
def train(train_loader, model, optimizer, epoch, total_step, writer, device, opt,
          log_every=100,
          rgb_stats=((0.485,0.456,0.406),(0.229,0.224,0.225)),
          th_stats=((0.5,0.5,0.5),(0.5,0.5,0.5))):
    model.train()
    size_rates = [1]
    loss_P1_record = AvgMeter()
    loss_P2_record = AvgMeter()

    global_step = (epoch - 1) * total_step

    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()

            # ---- data prepare ----
            images, thermals, gts = pack  # images: Bx3xHxW, thermals: Bx3xHxW, gts: Bx1xHxW
            images = images.to(device, non_blocking=True)
            thermals = thermals.to(device, non_blocking=True)
            gts = gts.to(device, non_blocking=True)

            # ---- rescale ----
            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:
                images = F.interpolate(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                thermals = F.interpolate(thermals, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                # Para GT usar nearest siempre
                gts = F.interpolate(gts, size=(trainsize, trainsize), mode='nearest')

            # ---- forward ----
            P1, P2 = model(images, thermals)  # P1: lista de predicciones, P2: pred principal

            # ---- loss function ----            
            #LOSS P1
            loss_P1 = 0.0
            gamma = 0.25
            normalizer = 0.0
            for it, out in enumerate(P1):
                w = 1.0 + gamma * it
                loss_P1 += w * structure_loss(out, gts)
                normalizer += w
            loss_P1 /= normalizer
            
            #LOSS P2
            loss_P2 = structure_loss(P2, gts)
            loss = loss_P1 + loss_P2
                        
            # ---- backward ----
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()

            # ---- recording loss ----
            if rate == 1:
                loss_P1_record.update(loss_P1.detach(), images.size(0))
                loss_P2_record.update(loss_P2.detach(), images.size(0))

        # ---- logging texto ----
        if i % 20 == 0 or i == total_step:
            msg = ('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                   'Loss P1: [{:.4f}], Loss P2: [{:.4f}]').format(
                datetime.now(), epoch, opt.epoch, i, total_step,
                loss_P1_record.show(), loss_P2_record.show())
            print(msg)
            logging.info(msg)

        # ---- visualizaciones periódicas ----
        global_step += 1
        if global_step % log_every == 0:
            # Usa la última pred de P1 reescalada a la GT para mostrar
            with torch.no_grad():
                pred_vis = P2
                pred_vis = torch.sigmoid(pred_vis)
                pred_vis = pred_vis.clamp(0, 1)
                log_train_images(writer, global_step, images, thermals, gts, pred_vis,
                                 rgb_stats=rgb_stats, th_stats=th_stats)

    # guardado periódico
    save_path = opt.save_path
    if epoch % opt.epoch_save == 0:
        torch.save(model.state_dict(), os.path.join(save_path, f'{epoch}_AGNet-PVT.pth'))


# =========================
# Main
# =========================
def load_matched_state_dict(model, state_dict, print_stats=True):
    num_matched, num_total = 0, 0
    curr_state_dict = model.state_dict()
    for key in curr_state_dict.keys():
        num_total += 1
        if key in state_dict and curr_state_dict[key].shape == state_dict[key].shape:
            curr_state_dict[key] = state_dict[key]
            num_matched += 1
    model.load_state_dict(curr_state_dict)
    if print_stats:
        print(f'Loaded state_dict: {num_matched}/{num_total} matched')


if __name__ == '__main__':
    #dataset = 'M3fd6'
    dataset = 'IguanaDataset'

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=200, help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    #parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='AdamW or SGD')
    parser.add_argument('--augmentation', default=True, help='random flip/rotation etc.')
    parser.add_argument('--batchsize', type=int, default=10, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=352, help='training image size (e.g., 352,704,1056)')
    parser.add_argument('--clip', type=float, default=1.0, help='gradient clipping margin')
    parser.add_argument('--load', type=str, default=None, help='checkpoint path to load')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='lr decay rate')
    parser.add_argument('--decay_epoch', type=int, default=100, help='lr decay every n epochs')
    parser.add_argument('--train_path', type=str, default=f'../../Datasets/{dataset}/train', help='train dataset path')
    parser.add_argument('--val_path', type=str, default=f'../../Datasets/{dataset}/val', help='val dataset path')
    parser.add_argument('--save_path', type=str, default=f'./model_pth/AGNet_{dataset}/')
    parser.add_argument('--epoch_save', type=int, default=1, help='save every n epochs')
    parser.add_argument('--log_every', type=int, default=5, help='steps between train visual logs')
    
    # Banderas para métricas de evaluación
    parser.add_argument('--use_mae', action='store_true', default=True, help='use MAE metric')
    parser.add_argument('--use_emeasure', action='store_true', default=True, help='use E-measure metric')
    parser.add_argument('--use_fmeasure', action='store_true', default=True, help='use F-measure metric')
    parser.add_argument('--use_smeasure', action='store_true', default=True, help='use S-measure metric')
    parser.add_argument('--use_wfmeasure', action='store_true', default=True, help='use Weighted F-measure metric')
    
    opt = parser.parse_args()

    os.makedirs(opt.save_path, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(opt.save_path, 'log.log'),
        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info("Network-Train")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    CE = torch.nn.BCEWithLogitsLoss()
    # ---- build model ----
    model = CamouflageDetectionNet().to(device)

    if opt.load is not None:
        pretrained_dict = torch.load(opt.load, map_location='cpu')
        print('!!!!!! Successfully load model from !!!!!! ', opt.load)
        load_matched_state_dict(model, pretrained_dict)

    print('model parameters', sum(p.numel() for p in model.parameters() if p.requires_grad))

    params = model.parameters()
    if opt.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params, opt.lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(params, opt.lr, weight_decay=1e-4, momentum=0.9)

    image_root = f'{opt.train_path}/Imgs/'
    gt_root = f'{opt.train_path}/GT/'
    thermal_root = f'{opt.train_path}/Thermal/'

    print("image_root: ", image_root)
    print("gt_root: ", gt_root)
    print("thermal_root: ", thermal_root)

    train_loader = get_loader(
        image_root, thermal_root, gt_root,
        batchsize=opt.batchsize, trainsize=opt.trainsize,
        augmentation=opt.augmentation
    )
    total_step = len(train_loader)

    writer = SummaryWriter(os.path.join(opt.save_path, 'summary'))

    print("#" * 20, "Start Training", "#" * 20)
    best_mae = 1
    best_epoch = 0

    #cosine_schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=opt.epoch, eta_min=1e-5)

    cosine_schedule = optim.lr_scheduler.CosineAnnealingWarmRestarts(
       optimizer,
       T_0=10,        # primer ciclo
       T_mult=2,      # cada ciclo dura el doble
       eta_min=1e-5
    )
        
    for epoch in range(1, opt.epoch+1):
        # schedule
        writer.add_scalar('train/learning_rate', cosine_schedule.get_last_lr()[0], global_step=epoch)
        logging.info('>>> current lr: {}'.format(cosine_schedule.get_last_lr()[0]))

        # train
        train(train_loader, model, optimizer, epoch, total_step, writer, device, opt,
              log_every=opt.log_every)

        # validation
        if epoch % opt.epoch_save == 0:
            val(model, epoch, opt.save_path, writer, opt, device)         
        
        cosine_schedule.step()        
 
