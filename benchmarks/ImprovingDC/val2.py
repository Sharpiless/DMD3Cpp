# CONFIG
import argparse

arg = argparse.ArgumentParser(description='depth completion')
arg.add_argument('-p', '--project_name', type=str, default='inference')
arg.add_argument('-c', '--configuration', type=str, default='val_LRRU_with_Three_DFU.yml')
arg.add_argument('-d', '--data_folder2', type=str, default='inference')
arg = arg.parse_args()
from configs import get as get_cfg
config = get_cfg(arg)

# ENVIRONMENT SETTINGS
import os
import cv2
rootPath = os.path.abspath(os.path.dirname(__file__))
import functools
if len(config.gpus) == 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpus[0])
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = functools.reduce(lambda x, y: str(x) + ',' + str(y), config.gpus)
import torch.nn.functional as F
# BASIC PACKAGES
import emoji
import time
import random
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader

# MODULES
from dataloaders.uni_loader import KittiDepth
from model import get as get_model
from summary import get as get_summary
from metric import get as get_metric
from utility import *

# VARIANCES
sample_, output_ = None, None
metric_txt_dir = None

# MINIMIZE RANDOMNESS
torch.manual_seed(config.seed)
torch.cuda.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)
np.random.seed(config.seed)
random.seed(config.seed)

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def test(args):

    # DATASET
    print(emoji.emojize('Prepare data... :writing_hand:', variant="emoji_type"), end=' ')
    global sample_, output_, metric_txt_dir
    data_test = KittiDepth(args.test_option, args, arg.data_folder2)
    loader_test = DataLoader(dataset=data_test,
                             batch_size=1,
                             shuffle=False,
                             num_workers=0)
    print('Done!')
    vis_dir = os.path.join("results", arg.data_folder2.split('/')[-1])
    os.makedirs(vis_dir, exist_ok=False)

    # NETWORK
    print(emoji.emojize('Prepare model... :writing_hand:', variant="emoji_type"), end=' ')
    model = get_model(args)
    net = model(args)
    net.cuda()
    print('Done!')
    total_params = count_parameters(net)

    # METRIC
    print(emoji.emojize('Prepare metric... :writing_hand:', variant="emoji_type"), end=' ')
    metric = get_metric(args)
    metric = metric(args)
    print('Done!')

    # SUMMARY
    print(emoji.emojize('Prepare summary... :writing_hand:', variant="emoji_type"), end=' ')
    summary = get_summary(args)
    try:
        if not os.path.isdir(args.test_dir):
            os.makedirs(args.test_dir)
        os.makedirs(args.test_dir, exist_ok=True)
        os.makedirs(args.test_dir + '/test', exist_ok=True)
        metric_txt_dir = os.path.join(args.test_dir + '/test/result_metric.txt')
        with open(metric_txt_dir, 'w') as f:
            f.write('test_model: {} \ntest_option: {} \nval:{} \ntest_name: {} \n'
                    'test_not_random_crop: {} \n'
                    'tta: {}\n \n'.format(args.test_model, args.test_option, 'val', args.test_name,
                                       args.test_not_random_crop,
                                       args.tta))
    except OSError:
        pass
    writer_test = summary(args.test_dir, 'test', args, None, metric.metric_name)
    print('Done!')

    # LOAD MODEL
    print(emoji.emojize('Load model... :writing_hand:', variant="emoji_type"), end=' ')
    if len(args.test_model) != 0:
        assert os.path.exists(args.test_model), \
            "file not found: {}".format(args.test_model)

        checkpoint_ = torch.load(args.test_model, map_location='cpu')
        model = remove_moudle(checkpoint_)
        key_m, key_u = net.load_state_dict(model, strict=False)

        if key_u:
            print('Unexpected keys :')
            print(key_u)

        if key_m:
            print('Missing keys :')
            print(key_m)

    net = nn.DataParallel(net)
    net.eval()
    print('Done!')

    num_sample = len(loader_test) * loader_test.batch_size
    pbar_ = tqdm(total=num_sample)
    t_total = 0
    with torch.no_grad():
        for batch_, sample_ in enumerate(loader_test):

            torch.cuda.synchronize()
            t0 = time.time()
            if args.tta:
                samplep = {key: val.cuda() for key, val in sample_.items()
                           if torch.is_tensor(val)}
                samplep['d_path'] = sample_['d_path']
                outputp = net(samplep)
                predp = outputp['results'][-1]

                samplef = {'dep': torch.flip(sample_['dep'], [-1]),
                           'rgb': torch.flip(sample_['rgb'], [-1]),
                           'ip': torch.flip(sample_['ip'], [-1]),
                           'dep_clear': torch.flip(sample_['dep_clear'], [-1])
                           }
                samplef = {key: val.cuda() for key, val in samplef.items()
                           if val is not None}
                outputf = net(samplef)
                predf = torch.flip(outputf['results'][-1], [-1])

                output_ = {'results': [(predp + predf) / 2.]}

            else:
                samplep = {key: val.float().cuda() for key, val in sample_.items()
                           if torch.is_tensor(val)}
                dep = samplep['dep']
                rgb = samplep['rgb']
                ip = samplep['ip']
                dep_clear = samplep['dep_clear']
                
                samplep['d_path'] = sample_['d_path']
                
                # 记录原始的高和宽，用于后续把 prediction 恢复回来
                orig_h, orig_w = rgb.shape[2], rgb.shape[3]
                
                # 2. 计算最接近的 64 的倍数大小
                # 使用 (x + 63) // 64 * 64 可以安全地向上取整到 64 的倍数
                new_h = (orig_h + 31) // 32 * 32
                new_w = (orig_w + 31) // 32 * 32
                if new_h != orig_h or new_w != new_w:
                
                    # 3. 对输入进行 Resize
                    # RGB 使用双线性插值 (Bilinear)，让图像更平滑
                    rgb_resized = F.interpolate(rgb, size=(new_h, new_w), mode='bilinear', align_corners=False)
                    # 稀疏深度图和 mask 必须严格使用最近邻插值 (Nearest)
                    dep_resized = F.interpolate(dep, size=(new_h, new_w), mode='nearest')
                    ip_resized = F.interpolate(ip, size=(new_h, new_w), mode='nearest')
                    dep_clear_resized = F.interpolate(dep_clear, size=(new_h, new_w), mode='nearest')
                    
                    samplep_resized = {
                        'rgb': rgb_resized, 'dep': dep_resized,
                        'ip': ip_resized, 'dep_clear': dep_clear_resized
                    }
                    output_ = net(samplep_resized)
                    pred_resized = output_['results'][-1]
                    pred = F.interpolate(pred_resized, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
                    output_['results'][-1] = pred
                else:
                    output_ = net(samplep)

            torch.cuda.synchronize()
            t1 = time.time()
            t_total += (t1 - t0)
            if 'test' not in args.test_option:
                metric_test = metric.evaluate(output_['results'][-1], samplep['gt'], 'test')
            else:
                metric_test = metric.evaluate(output_['results'][-1], samplep['dep'], 'test')
            
            # import IPython
            # IPython.embed()
            # exit()
            # tensor([[5.5874e-01, 1.3622e-01, 9.9910e-04, 5.1132e-04, 6.7578e-03, 9.9919e-01,
            #  9.9990e-01, 9.9992e-01]], device='cuda:0')

            depth_validpoint_number = count_validpoint(samplep['dep'])
            depth_validpoint_number_clear = count_validpoint(samplep['dep'])
            with open(metric_txt_dir, 'a') as f:
                f.write('{}; RMSE:{}; MAE:{}; vp_pre:{}; vp_post:{}\n'.format(samplep['d_path'][0].split('/')[-1],
                                                                         metric_test.data.cpu().numpy()[0, 0]*1000,
                                                                         metric_test.data.cpu().numpy()[0, 1]*1000,
                                                                         depth_validpoint_number,
                                                                         depth_validpoint_number_clear))

            writer_test.add(None, metric_test)
            if args.save_test_image:
                writer_test.save(args.epochs, batch_, samplep, output_)

            current_time = time.strftime('%y%m%d@%H:%M:%S')
            error_str = '{} | Test'.format(current_time)
            pbar_.set_description(error_str)
            pbar_.update(loader_test.batch_size)
            
            if True:
                pred_np = output_['results'][-1].squeeze().detach().cpu().numpy().astype(np.float32)

                pred_min, pred_max = pred_np.min(), pred_np.max()
                pred_vis = (pred_np - pred_min) / (pred_max - pred_min + 1e-8) * 255
                pred_vis = pred_vis.astype(np.uint8)
                pred_vis = cv2.applyColorMap(pred_vis, cv2.COLORMAP_JET)
                cv2.imwrite(os.path.join(vis_dir ,f"{batch_:08d}_depth.png"), pred_vis)


    pbar_.close()
    _ = writer_test.update(args.epochs, samplep, output_,
                           online_loss=False, online_metric=False, online_rmse_only=False, online_img=False)
    t_avg = t_total / num_sample
    with open(metric_txt_dir, 'a') as f:
        f.write('Elapsed time : {} sec, '
          'Average processing time : {} sec'.format(t_total, t_avg))

    print('Elapsed time : {} sec, '
          'Average processing time : {} sec'.format(t_total, t_avg))


if __name__ == '__main__':
    test(config)
