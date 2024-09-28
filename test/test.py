import time
import numpy as np
import torch
import os
import sys
from tqdm import tqdm
from dataset import collate_fn, CorrespondencesDataset
from utils import compute_pose_error, pose_auc, estimate_pose_norm_kpts, estimate_pose_from_E
from config import get_config

sys.path.append('../core')
from mfanet import MFANet

torch.set_grad_enabled(False)
torch.manual_seed(0)


def batch_episym(x1, x2, F):
    batch_size, num_pts = x1.shape[0], x1.shape[1]
    x1 = torch.cat([x1, x1.new_ones(batch_size, num_pts, 1)], dim=-1).reshape(batch_size, num_pts, 3, 1)
    x2 = torch.cat([x2, x2.new_ones(batch_size, num_pts, 1)], dim=-1).reshape(batch_size, num_pts, 3, 1)
    F = F.reshape(-1, 1, 3, 3).repeat(1, num_pts, 1, 1)
    x2Fx1 = torch.matmul(x2.transpose(2, 3), torch.matmul(F, x1)).reshape(batch_size, num_pts)
    Fx1 = torch.matmul(F, x1).reshape(batch_size, num_pts, 3)
    Ftx2 = torch.matmul(F.transpose(2, 3), x2).reshape(batch_size, num_pts, 3)
    ys = x2Fx1 ** 2 * (
            1.0 / (Fx1[:, :, 0] ** 2 + Fx1[:, :, 1] ** 2 + 1e-15) +
            1.0 / (Ftx2[:, :, 0] ** 2 + Ftx2[:, :, 1] ** 2 + 1e-15))
    return ys


def inlier_test(config, polar_dis, inlier_mask):
    polar_dis = polar_dis.reshape(inlier_mask.shape).unsqueeze(0)
    inlier_mask = torch.from_numpy(inlier_mask).type(torch.float32)
    is_pos = (polar_dis < config.obj_geod_th).type(inlier_mask.type())
    is_neg = (polar_dis >= config.obj_geod_th).type(inlier_mask.type())
    precision = torch.mean(
        torch.sum(inlier_mask * is_pos, dim=1) /
        (torch.sum(inlier_mask * (is_pos + is_neg), dim=1) + 1e-15)
    )
    recall = torch.mean(
        torch.sum(inlier_mask * is_pos, dim=1) /
        torch.sum(is_pos, dim=1)
    )
    f_scores = 2 * precision * recall / (precision + recall + 1e-15)

    return precision, recall, f_scores


def count_flops(model, input_tensor):
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        model(input_tensor)
    return prof.key_averages().total_average().cpu_time_total


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def test(config, use_essential=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Running inference on device \"{}\"'.format(device))

    test_dataset = CorrespondencesDataset(config.data_te, config)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=8, pin_memory=True, collate_fn=collate_fn)

    # model_config = torch.load(os.path.join(config.model_file, 'config.th'))
    # config.grid_nums = model_config.grid_nums
    # config.sample_rates = model_config.sample_rates
    # try:
    #     config.up_sample = model_config.up_sample
    # except AttributeError:
    #     config.up_sample = False
    model = MFANet(config)

    save_file_best = os.path.join(config.model_file, config.model_name)
    print("load: %s" % config.model_name)
    if not os.path.exists(save_file_best):
        print("Model File {} does not exist! Quiting".format(save_file_best))
        exit(1)
    # Restore model
    checkpoint = torch.load(save_file_best)
    model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()})
    model.cuda()
    model.eval()

    err_ts, err_Rs = [], []
    precision_all, recall_all, f_scores_all = [], [], []

    total_time = 0.0
    num_images = len(test_loader)
    max_memory = 0.0

    # Calculate Flops and Params
    params = count_params(model)
    print('Model Parameters: {:.6f} million'.format(params / 1e6))

    for index, test_data in enumerate(tqdm(test_loader)):
        x = test_data['xs'].to(device)
        y = test_data['ys'].to(device)
        R_gt, t_gt = test_data['Rs'], test_data['ts']

        data = {}
        data['xs'] = x
        start_time = time.time()
        logits_list, e_hat_list = model(data)
        end_time = time.time()
        total_time += (end_time - start_time)
        torch.cuda.synchronize()
        max_memory = max(max_memory, torch.cuda.max_memory_allocated(device) / (1024 ** 2))  # in MB

        logits = logits_list[-1]
        e_hat = e_hat_list[-1].cpu().detach().numpy().reshape(3, 3)

        mkpts0 = x.squeeze()[:, :2].cpu().detach().numpy()
        mkpts1 = x.squeeze()[:, 2:].cpu().detach().numpy()

        # use essential matrix
        if use_essential:
            y_hat = batch_episym(x[:, 0, :, :2], x[:, 0, :, 2:], e_hat_list[-1])
            inlier_mask = y_hat.squeeze().cpu().detach().numpy() < config.obj_geod_th
        # use logits
        else:
            inlier_mask = logits.squeeze().cpu().detach().numpy() > config.inlier_threshold

        mask_kp0 = mkpts0[inlier_mask]
        mask_kp1 = mkpts1[inlier_mask]

        if config.use_ransac == True:
            ret = estimate_pose_norm_kpts(mask_kp0, mask_kp1, conf=config.ransac_prob)
        else:
            if e_hat.shape[0] == 0:
                print("Algorithm has no essential matrix output, can not eval without robust estimator such as RANSAC.")
                print("Try to set use_ransac=True in config file.")
                exit(1)
            ret = estimate_pose_from_E(mkpts0, mkpts1, inlier_mask, e_hat)
        if ret is None:
            err_t, err_R = np.inf, np.inf
            precision_all.append(torch.zeros(1, )[0])
            recall_all.append(torch.zeros(1, )[0])
            f_scores_all.append(torch.zeros(1, )[0])
        else:
            R, t, inlier_mask_new = ret
            T_0to1 = torch.cat([R_gt.squeeze(), t_gt.squeeze().unsqueeze(-1)], dim=-1).numpy()
            err_t, err_R = compute_pose_error(T_0to1, R, t)

        err_ts.append(err_t)
        err_Rs.append(err_R)

        precision, recall, f_scores = inlier_test(config, y, inlier_mask)

        precision_all.append(precision.cpu())
        recall_all.append(recall.cpu())
        f_scores_all.append(f_scores.cpu())

    avg_time = total_time / num_images
    out_eval = {'error_t': err_ts,
                'error_R': err_Rs}

    pose_errors = []
    for idx in range(len(out_eval['error_t'])):
        pose_error = np.maximum(out_eval['error_t'][idx], out_eval['error_R'][idx])
        pose_errors.append(pose_error)

    thresholds = [5, 10, 20]
    aucs = pose_auc(pose_errors, thresholds)
    aucs = [100. * yy for yy in aucs]
    precision, recall, f_scores = np.mean(np.asarray(precision_all)) * 100, np.mean(
        np.asarray(recall_all)) * 100, np.mean(np.asarray(f_scores_all)) * 100

    print('Evaluation Results {} RANSAC (mean over {} pairs):'
          .format("with" if config.use_ransac else "without", len(test_loader)))
    print('AUC@5\t AUC@10\t AUC@20\t')
    print('{:.2f}\t {:.2f}\t {:.2f}\t'.format(aucs[0], aucs[1], aucs[2]))
    print('Prec\t Rec\t F1\t')
    print('{:.2f}\t {:.2f}\t {:.2f}\t'.format(precision, recall, f_scores))

    print('\nAdditional Metrics:')
    print('Average Inference Time per Image: {:.6f} seconds'.format(avg_time))
    print('Peak Memory Usage: {:.2f} MB'.format(max_memory))

    return aucs, precision, recall


if __name__ == '__main__':
    config, unparsed = get_config()
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id
    use_essential = False
    config.use_ransac = False
    print("use ransac: {}".format(config.use_ransac))
    test(config, use_essential)
    config.use_ransac = True
    print("use ransac: {}".format(config.use_ransac))
    test(config)

    # model = MFANet(config)
    # save_file_best = os.path.join(config.model_file, config.model_name)
    # checkpoint = torch.load(save_file_best)
    # new_state_dict = {k.replace('unpools.', 'adj.'): v for k, v in checkpoint['state_dict'].items()}
    # new_state_dict = {k.replace('unpool.', 'att.'): v for k, v in new_state_dict.items()}
    # checkpoint['state_dict'] = new_state_dict
    # model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()})
    # torch.save({
    #     'epoch': checkpoint['epoch'],
    #     'state_dict': model.state_dict(),
    #     'best_acc': checkpoint['best_acc'],
    #     'optimizer': checkpoint['optimizer'],
    # }, save_file_best)
    #
    # print()
