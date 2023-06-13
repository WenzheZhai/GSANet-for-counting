import logging
import time

import numpy as np
import scipy
import scipy.io as io
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
from tqdm import tqdm


class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        msg = self.format(record)
        tqdm.write(msg)


def get_logger(name="Train", save_path="./results/train.log", mode="w"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    tqdm_handler = TqdmLoggingHandler()
    file_handler = logging.FileHandler(save_path, mode)

    logger.addHandler(tqdm_handler)
    logger.addHandler(file_handler)

    logger.info("-" * 25 + f" {name} " + "-" * 25)

    return logger


def output():
    print("utils_me install successfully")
    # 自定义时间字符串
    print(time.strftime("%Y-%m-%d %H:%M:%S"))


def np2jpg(img_np, save_path):
    height, width = img_np.shape
    print(height, width)

    figure, axes = plt.subplots()
    axes.imshow(img_np, cmap="jet")
    plt.axis("off")

    figure.set_size_inches(width / 400, height / 400)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)

    plt.savefig(save_path, dpi=400)
    print(save_path)
    plt.close()


def density_map(img_path="", gt_path="", sigma=0):
    """
    sigma=0 -> density = gaussian_filter_density(k)\n
    sigma=3 -> density = gaussian_filter(k, sigma)\n
    density -> <class 'numpy.ndarray'>\n
    h5py -> gt = np.asarray(f['density'])
    """
    img = plt.imread(img_path)
    # plt.imsave('', img) 3通道绘制
    # np2jpg 单通道绘制
    k = np.zeros((img.shape[0], img.shape[1]))  # H W

    if not gt_path:
        gt_path = (
            img_path.replace("images", "ground_truth")
            .replace("IMG_", "GT_IMG_")
            .replace("jpg", "mat")
        )
    gt = io.loadmat(gt_path)
    gt = gt["image_info"][0, 0][0, 0][0]  # (x, y)

    for i in range(len(gt)):
        # 人头的纵坐标小于图像的高度 人头的横坐标小于图像的宽度
        if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
            # point-level annotation 人头的像素值为1
            k[int(gt[i][1]), int(gt[i][0])] = 1

    if sigma:
        density = gaussian_filter(k, sigma)
    else:
        density = gaussian_filter_density(k)

    return density


def gaussian_filter_density(gt):
    print(gt.shape)
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density

    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))
    leafsize = 2048
    # build kdtree
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(pts, k=4)

    print("generate density...")
    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1], pt[0]] = 1.0
        if gt_count > 1:
            sigma = (distances[i][1] + distances[i][2] + distances[i][3]) * 0.1
        else:
            sigma = np.average(np.array(gt.shape)) / 2.0 / 2.0  # case: 1 point
        density += gaussian_filter(pt2d, sigma, mode="constant")
    print("done")
    return density


class AverageMeter(object):
    """记录和更新变量(变量管理 相当于创建一个可管理的变量)

    Args:
        val: 变量当前值
        sum: 变量累加值
        count: 变量累加次数
        avg: 变量平均值
    """

    def __init__(self):
        """创建一个可管理变量"""
        self.reset()

    def reset(self):
        """所有值清零"""
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0

    def update(self, val):
        """更新变量值 即给变量重新赋值

        Args:
            val (float or int): 变量更新值
        """
        self.val = val
        self.sum = self.sum + self.val
        self.count += 1
        self.avg = self.sum / self.count


class Timer(object):
    """计时器

    Args:
        val: 本次计时时间
        sum: 总计时时间
        count: 计时次数
        avg: 平均计时时间
    """

    def __init__(self):
        """创建计时器"""
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0

    def tic(self):
        """开始计时"""
        self.val = time.time()

    def toc(self):
        """停止计时"""
        self.val = time.time() - self.val
        self.sum = self.sum + self.val
        self.count += 1
        self.avg = self.sum / self.count
