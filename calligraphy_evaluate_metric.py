# coding=utf-8
import argparse
import glob
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import shutil
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
# from matplotlib import pyplot as plt
from sklearn import metrics, neighbors
from pytorch_fid import fid_score
from torchvision.transforms import transforms
import json
from PIL import Image
import torch
import lpips
from tabulate import tabulate
# import wcwidth
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from operator import truediv

# 加载预训练的LPIPS模型
lpips_model = lpips.LPIPS(net="alex").to("cuda:0")


def read_image(path):
    img = np.array(Image.open(path).convert('RGB').resize((256, 256)))
    return img


# def read_image(path):
#     mat = plt.imread(path)
#     side = int(mat.shape[1] / 2)
#     assert side * 2 == mat.shape[1]
#     img_A = mat[:, :side]  # target
#     img_B = mat[:, side:]  # source
#
#     return img_A, img_B

# def get_mse(img_1, img_2):
#     return np.mean((img_1 - img_2) ** 2)

def get_ssim(img_1, img_2):
    return ssim(img_1, img_2, multichannel=True, channel_axis=2)


def get_iou(img_1, img_2):
    cal_outputs = img_1.flatten()
    label = img_2.flatten()
    c = metrics.confusion_matrix(cal_outputs, label)
    FP = c.sum(axis=0) - np.diag(c)
    FN = c.sum(axis=1) - np.diag(c)
    TP = np.diag(c)
    TN = c.sum() - (FP + FN + TP)
    MIou = (TP / (TP + FP + FN) + TN / (TN + FN + FP)) / 2
    mean_iou = np.mean(MIou)
    return mean_iou


def get_psnr(img_1, img_2):
    return psnr(img_1, img_2)


def get_lpips(img_1, img_2):
    image1_tensor = torch.tensor(img_1).permute(2, 0, 1).unsqueeze(0).float().to("cuda:0") / 255.0
    image2_tensor = torch.tensor(img_2).permute(2, 0, 1).unsqueeze(0).float().to("cuda:0") / 255.0
    return lpips_model(image1_tensor, image2_tensor).squeeze().float()


def get_average_mse_ssim_psnr(fig_names_1, fig_names_2):
    ssim_total = 0
    psnr_total = 0
    lpips_total = 0
    l = len(fig_names_2)
    for i in range(len(fig_names_2)):
        img_1 = read_image(fig_names_1[i])
        img_2 = read_image(fig_names_2[i])
        ssim_total += get_ssim(img_1, img_2)
        psnr_total += get_psnr(img_1, img_2)
        lpips_total += get_lpips(img_1, img_2)

    ssim_avg = ssim_total / l
    psnr_avg = psnr_total / l
    lpips_avg = lpips_total / l
    return ssim_avg, psnr_avg, lpips_avg


# def get_average_mse_ssim_psnr(fig_names):
#     mse_total = 0
#     ssim_total = 0
#     psnr_total = 0
#     for i in range(len(fig_names)):
#         print(fig_names[i])
#         img_1,img_2 = read_image(fig_names[i])
#         # img_1 = img_1[:,:,0]
#         # img_2 = img_2[:,:,0]
#         mse_total += get_mse(img_1, img_2)
#         ssim_total += get_ssim(img_1, img_2)
#         psnr_total += get_psnr(img_1, img_2)
#     mse_avg = mse_total / (i + 1)
#     ssim_avg = ssim_total / (i + 1)
#     psnr_avg = psnr_total / (i + 1)
#     return mse_avg, ssim_avg, psnr_avg


def main(args, phase):
    total_ssim, total_psnr, total_fid, total_lpips = 0, 0, 0, 0

    all_style = os.listdir(args.g)
    print(f'start to test for {phase}')
    style_bar = tqdm(all_style, desc="calculate index for every style", unit='style')

    for style_name in style_bar:

        img_gt_paths = []
        img_pred_paths = []
        for gt_path in glob.glob(os.path.join(args.g, style_name, '*.jpg')):
            img_gt_paths.append(gt_path)

        for pt_path in glob.glob(os.path.join(args.p, style_name, '*.jpg')):
            img_pred_paths.append(pt_path)

        img_gt_paths.sort()
        img_pred_paths.sort()

        ssim, psnr, lpips = get_average_mse_ssim_psnr(img_gt_paths, img_pred_paths)
        # mse, ssim, psnr = get_average_mse_ssim_psnr(img_pred_paths)
        fid = fid_score.calculate_fid_given_paths([os.path.join(args.g, style_name), os.path.join(args.p, style_name)],
                                                  batch_size=1, dims=64, device='cuda:0', num_workers=0)

        total_ssim += ssim
        total_psnr += psnr
        total_fid += fid
        total_lpips += lpips

    print("Average SSIM for total styles: {:.5f}".format(total_ssim / len(all_style)))
    print("Average PSNR for total styles: {:.5f}".format(total_psnr / len(all_style)))
    print("Average FID for total styles: {:.5f}".format(total_fid / len(all_style)))
    print("Average LPIPS for total styles: {:.5f}".format(total_lpips / len(all_style)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", default=r'outputs/FontDiffuser/sample', help="Path to ground truth folder")
    parser.add_argument("-p", default=r'outputs/FontDiffuser/sample',
                        help="Path to prediction folder")
    args = parser.parse_args()
    gt_ls = ['sfuc', 'ufsc', 'ufuc']
    gt = args.g
    pre = args.p
    for i in range(3):
        args.g = os.path.join(gt, gt_ls[i], "real_images")
        args.p = os.path.join(pre, gt_ls[i], "fake_images")
        main(args, gt_ls[i])

# calculate mse, ssim score for each folder
# calculate total_mse, total_ssim score for all folders
#     test_char = ['聖', '甾', '分', '文', '代', '波', '沙', '上', '詹', '節', '塗', '織', '其', '鳥', '拔', '慧', '業', '堅', '然', '缺', '爺', '巨', '歹', '鷲', '共', '致', '鹼', '寒', '梵', '揚', '垂', '自', '旰', '鼱', '斤', '道', '扮', '貞', '舉', '猶', '鼾', '詔', '倜', '穀', '勾', '故', '永', '販', '滾', '露', '治', '河', '淨', '夫', '揀', '臼', '響', '麒', '序', '夾', '為', '群', '革', '劫', '灰', '跟', '梁', '琮', '青', '西', '燕', '端', '品', '豈', '功', '齣', '潮', '氏', '或', '琯', '途', '空', '且', '珠', '十', '丑', '妙', '朝', '槃', '刀', '歲', '旨', '識', '占', '能', '深', '目', '躬', '之', '羊', '顯', '龔', '製', '乘', '重', '宇', '通', '伏', '張', '是', '停', '乾', '婜', '貌', '前', '教', '女', '百', '蠢', '敷', '記', '舌', '兩', '禺', '溪', '時', '給', '乃', '學', '遐', '帝', '扶', '物', '吳', '句', '于', '奄', '惑', '還', '立', '皮', '寂', '國', '爸', '遠', '蜊', '正', '臻', '發', '昌', '山', '及', '宅', '娓', '幽', '雨', '達', '含', '塔', '昭', '廣', '謀', '嘰', '盂', '雅', '滅', '盛', '鵣', '蛩', '潛', '劈', '苞', '幅', '齦', '厹', '飛', '中', '雙', '馳', '酊', '衡', '龕', '流', '宣', '也', '宗', '風', '成', '年', '谷', '管', '鼩', '戎', '平', '皎', '開', '忘', '免', '騰', '誰', '黏', '我', '會', '度', '溼', '癸', '火', '近', '斛', '養', '經', '天', '良', '先', '師', '惟', '非', '眾', '團', '載', '要', '仰', '接', '引', '鼯', '幾', '匍', '春', '恒', '爹', '窮', '下', '論', '民', '赫', '若', '戶', '尋', '兌', '福', '埋', '善', '七', '待', '烷', '儀', '兔', '套', '建', '庸', '玄', '測', '同', '吾', '鼬', '壬', '勝', '慈', '亭', '醉', '精', '門', '察', '廷', '計', '囚', '寺', '朗', '土', '褚', '聯', '苯', '禮', '石', '乎', '部', '心', '褂', '與', '動', '封', '遊', '感', '未', '書', '琰', '林', '邀', '彼', '超', '四', '奇', '源', '輕', '曹', '花', '有', '老', '大', '域', '傅', '桂', '海', '陰', '真', '豕', '萬', '殺', '御', '栽', '偏', '焰', '性', '長', '聞', '呂', '化', '一', '勞', '洩', '亡', '嶺', '飄', '臣', '跡', '謂', '名', '朔', '照', '弘', '譯', '昇', '室', '明', '凡', '懷', '券', '祀', '歷', '清', '者', '則', '令', '葉', '沈', '人', '嘗', '將', '朽', '奸', '五', '陳', '法', '資', '昏', '睢', '不', '固', '神', '緣', '點', '遂', '易', '拯', '而', '己', '華', '奧', '雪', '艮', '晨', '次', '咳', '阻', '峻', '惡', '足', '生', '墜', '謄', '求', '極', '町', '以', '崇', '綱', '歸', '出', '孝', '琺', '辛', '蓋', '影', '乂', '慶', '無', '遵', '被', '賢', '異', '日', '從', '典', '象', '誠', '助', '鹿', '區', '言', '常', '難', '佳', '臨', '字', '比', '并', '迷', '古', '義', '韋', '三', '德', '塵', '勺', '二', '微', '莫', '況', '唔', '月', '澆', '莉', '愚', '齟', '瓶', '材', '虎', '知', '際', '守', '在', '舳', '吃', '欅', '壯', '高', '鈍', '撲', '囊', '理', '水', '於', '虫', '蓮', '卜', '揖', '累', '智', '往', '川', '潤', '定', '陽', '斯', '旅', '世', '全', '責', '千', '鼙', '灣', '地', '咨', '六', '唐', '政', '冷', '隹', '所', '欠', '質', '境', '秀', '斜', '台', '峰', '類', '沒', '相', '恕', '棒', '形', '家', '機', '冬', '舀', '情', '凝', '俎', '茲', '顜', '徽', '來', '剄', '金', '皇', '光', '東', '酋', '今', '噓', '取', '針', '殘', '俯', '體', '藏', '般', '雲', '衕', '暑', '夕']
#
#
#     with open(r'E:\code\synthesis\WordStylist-main\dataset_1\tr_va.gt.filter', "r") as f:
#         data = f.read()
#     data = data.split('\n')
#     with open('E:\code\synthesis\WordStylist-main\dataset_1\writers_dict.json', "r") as f:
#         wr_dict = json.load(f)
#     test_data = []
#     for i in data:
#         # print(i)
#         c = i.split(';')
#         s = wr_dict[c[0]]
#         if c[2] in test_char and s == 0:
#             path_a = os.path.join(r'E:\code\synthesis\WordStylist-main\dataset_1\gt\train\sty_0', i.split(';')[1])
#             path_b = os.path.join(r'E:\code\synthesis\WordStylist-main\dataset_1\xx\sty_0', i.split(';')[1])
#             if not os.path.exists(path_a):
#                 print(path_a)
#                 continue
#             shutil.copy(path_a,path_b)
#
