import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from datetime import datetime

from utils.dataloader import My_test_dataset  # Debe retornar (image, thermal, gt, name)
from lib.cod_net import CamouflageDetectionNet
from py_sod_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure  

def resize_to_match(img1, img2):  
    """Redimensiona img2 para que coincida con las dimensiones de img1"""  
    if img1.shape != img2.shape:  
        return cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_NEAREST)  
    return img2  
    
def denorm(img, mean, std):
    """
    img: BxCxHxW (float), mean/std: tuplas por canal
    retorna img en [0,1]
    """
    device = img.device
    mean = torch.tensor(mean, device=device).view(1, -1, 1, 1)
    std = torch.tensor(std, device=device).view(1, -1, 1, 1)
    x = img * std + mean
    return x.clamp(0.0, 1.0)


def save_gray(path, tensor_1x1xHxW):
    arr = (tensor_1x1xHxW[0, 0].cpu().numpy() * 255.0).astype(np.uint8)
    cv2.imwrite(path, arr)


def make_vis_grid(image, thermal, gt, pred,
                  rgb_stats=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                  th_stats=((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))):
    """
    Devuelve una imagen BGR concatenada horizontalmente: [RGB | Thermal | GT | Pred]
    image, thermal: 1x3xHxW normalizados
    gt, pred: 1x1xHxW en [0,1]
    """
    rgb_mean, rgb_std = rgb_stats
    th_mean, th_std = th_stats

    img_vis = denorm(image, rgb_mean, rgb_std)[0].permute(1, 2, 0).cpu().numpy()  # HxWx3 en [0,1]
    th_vis = denorm(thermal, th_mean, th_std)[0].permute(1, 2, 0).cpu().numpy()   # HxWx3 en [0,1]
    gt_vis = (gt[0, 0].cpu().numpy() * 255).astype(np.uint8)                       # HxW
    pred_vis = (pred[0, 0].cpu().numpy() * 255).astype(np.uint8)                   # HxW

    rgb_bgr = (img_vis * 255).astype(np.uint8)[:, :, ::-1]
    th_bgr = (th_vis * 255).astype(np.uint8)[:, :, ::-1]
    gt_bgr = cv2.cvtColor(gt_vis, cv2.COLOR_GRAY2BGR)
    pred_bgr = cv2.cvtColor(pred_vis, cv2.COLOR_GRAY2BGR)

    grid = np.concatenate([rgb_bgr, th_bgr, gt_bgr, pred_bgr], axis=1)
    return grid

def main():
    for idx in range(1,201):    
        parser = argparse.ArgumentParser()
        parser.add_argument('--testsize', type=int, default=416, help='testing size (debe coincidir con train)')
        parser.add_argument('--pth_path', type=str, default=f'C:/Respaldo/Henry/Proyecto Camuflaje/Codigo/AGNet-main/model_pth/AGNet_IguanaDataset/{idx}_AGNet-PVT.pth')
        parser.add_argument('--dataset', type=str, default='M3FD')
        parser.add_argument('--data_root', type=str, default='../../Datasets', help='raíz de datasets (coincidir mayúsculas/minúsculas)')
        parser.add_argument('--save_vis', action='store_true', default=True, help='guardar grid RGB|Thermal|GT|Pred')
        parser.add_argument('--vis_subdir', type=str, default='vis', help='subcarpeta para guardar visualizaciones')
        args = parser.parse_args()
        
        print("*"*25)
        print(f"IDX: {idx}, Path: {args.pth_path}")
        
        
        FM = Fmeasure()  
        WFM = WeightedFmeasure()  
        SM = Smeasure()  
        EM = Emeasure()  
        M = MAE()  
        
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        data_path = os.path.join(args.data_root, args.dataset, 'test')
        save_dir_root = f'./results_/{os.path.basename(os.path.dirname(args.pth_path))}_{idx}/{args.dataset}'
        os.makedirs(save_dir_root, exist_ok=True)
        if args.save_vis:
            save_vis_dir = os.path.join(save_dir_root, args.vis_subdir)
            os.makedirs(save_vis_dir, exist_ok=True)

        # Cargar modelo
        model = CamouflageDetectionNet().to(device)
        state = torch.load(args.pth_path, map_location=device)
        try:
            model.load_state_dict(state, strict=True)
        except Exception as e:
            print(f'[WARN] load_state_dict strict=True falló: {e}')
            model.load_state_dict(state, strict=False)
        model.eval()

        # Rutas de test
        image_root = os.path.join(data_path, 'Imgs') + '/'
        gt_root = os.path.join(data_path, 'GT') + '/'
        thermal_root = os.path.join(data_path, 'Thermal') + '/'
        print('Test roots:', image_root, gt_root, thermal_root)

        # Dataset de test: debe devolver (image, thermal, gt, name)
        test_loader = My_test_dataset(
            image_root=image_root,
            thermal_root=thermal_root,   # OJO: orden correcto
            gt_root=gt_root,
            testsize=args.testsize
        )
        print('Total test samples:', test_loader.size)

        total_mae = 0.0
        for i in range(test_loader.size):
            with torch.no_grad():
                image, thermal, gt, name = test_loader.load_data()  # image/thermal: 1x3xHxW, gt: 1x1xHxW
                # A GPU
                image = image.to(device, non_blocking=True)
                thermal = thermal.to(device, non_blocking=True)
                gt = gt.to(device, non_blocking=True)

                # Forward
                outs_list, out_single = model(image, thermal)

                # Reescalar a tamaño de GT
                target_size = gt.shape[-2:]
                combined = 0.0
                if isinstance(outs_list, (list, tuple)) and len(outs_list) > 0:
                    for o in outs_list:
                        combined = combined + F.interpolate(o, size=target_size, mode='bilinear', align_corners=False)
                if out_single is not None:
                    combined = combined + F.interpolate(out_single, size=target_size, mode='bilinear', align_corners=False)

                pred = torch.sigmoid(combined).clamp(0, 1)  # 1x1xHxW

                # MAE por imagen
                mae = torch.mean(torch.abs(pred - gt)).item()
                total_mae += mae

                # Guardar pred como PNG 8-bit
                save_path = os.path.join(save_dir_root, name)
                save_gray(save_path, pred)
                print(f'[{i+1}/{test_loader.size}] {args.dataset} - {name} | MAE: {mae:.6f}')

                
                #print(os.path.join(gt_root, name))
                #print(save_path)
                
                #gt_ = cv2.imread(os.path.join(gt_root, name), cv2.IMREAD_GRAYSCALE)
                #pred_ = cv2.imread(save_path, cv2.IMREAD_GRAYSCALE)

                gt_np = (gt.squeeze().cpu().numpy() * 255).astype(np.uint8)        # HxW
                pred_np = (pred.squeeze().cpu().numpy() * 255).astype(np.uint8)    # HxW
                
                # Redimensionar pred para que coincida con mask  
                #pred_np = resize_to_match(gt_np, pred_np)  
                gt_np = resize_to_match(pred_np, gt_np)  
                                
                # Verificar que ambas imágenes tengan el mismo tamaño  
                if gt_np.shape != pred_np.shape:  
                    print(f"Error: Las dimensiones no coinciden después del resize para {base_name}")  
                    continue  
                    
                # Calcular métricas para la imagen actual  
                FM.step(pred=pred_np, gt=gt_np)  
                WFM.step(pred=pred_np, gt=gt_np)  
                SM.step(pred=pred_np, gt=gt_np)  
                EM.step(pred=pred_np, gt=gt_np)  
                M.step(pred=pred_np, gt=gt_np)  

                # Obtener resultados para la imagen actual  
                #fm = FM.get_results()["fm"]  
                #wfm = WFM.get_results()["wfm"]  
                #sm = SM.get_results()["sm"]  
                #em = EM.get_results()["em"]  
                #mae = M.get_results()["mae"]  
                
                
                # Visualización opcional
                if args.save_vis:
                    grid = make_vis_grid(image, thermal, gt, pred)
                    cv2.imwrite(os.path.join(save_vis_dir, name), grid)

        avg_mae = total_mae / max(1, test_loader.size)
        print(f'[{datetime.now()}] AVG MAE on {args.dataset}: {avg_mae:.6f}')
        # También puedes guardar a un txt si quieres:
        with open(os.path.join(save_dir_root, 'metrics.txt'), 'w') as f:
            f.write(f'AVG_MAE={avg_mae:.6f}\n')
            
        
        # Calcular métricas globales  
        fm = FM.get_results()["fm"]  
        wfm = WFM.get_results()["wfm"]  
        sm = SM.get_results()["sm"]  
        em = EM.get_results()["em"]  
        mae = M.get_results()["mae"]  

        results = {  
            "Smeasure": sm,  
            "wFmeasure": wfm,  
            "MAE": mae,  
            "adpEm": em["adp"],  
            "meanEm": em["curve"].mean(),  
            "maxEm": em["curve"].max(),  
            "adpFm": fm["adp"],  
            "meanFm": fm["curve"].mean(),  
            "maxFm": fm["curve"].max(),  
        }  
        print(results) 
        
        
        with open("C:/Respaldo/Henry/Proyecto Camuflaje/Codigo/AGNet-main/evalresults.txt", "a") as file:  
            file.write(f"{idx}_AGNet-PVT {str(results)}\n")  


if __name__ == '__main__':
    main()
