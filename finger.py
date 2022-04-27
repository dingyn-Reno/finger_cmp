# pip install opencv-python
import math

import cv2
# pip install scikit-image
# pip install numpy
import numpy as np

_sigma_conv = (3.0 / 2.0) / ((6 * math.log(10)) ** 0.5)
mcc_sigma_s = 7.0
mcc_tau_psi = 400.0
mcc_mu_psi = 1e-2


def Gs(t_sqr):
    """Gaussian function with zero mean and mcc_sigma_s standard deviation, see eq. (7) in MCC paper"""
    return np.exp(-0.5 * t_sqr / (mcc_sigma_s ** 2)) / (math.tau ** 0.5 * mcc_sigma_s)


def Psi(v):
    """Sigmoid function that limits the contribution of dense minutiae clusters, see eq. (4)-(5) in MCC paper"""
    return 1. / (1. + np.exp(-mcc_tau_psi * (v - mcc_mu_psi)))


def _gabor_sigma(ridge_period):
    return _sigma_conv * ridge_period


def gabor_kernel(period, orientation):
    f = cv2.getGaborKernel(_gabor_size(period), _gabor_sigma(period), np.pi / 2 - orientation, period, gamma=1, psi=0)
    f /= f.sum()
    f -= f.mean()
    return f


def _gabor_size(ridge_period):
    p = int(round(ridge_period * 2 + 1))
    if p % 2 == 0:
        p += 1
    return (p, p)


def compute_crossing_number(values):
    return np.count_nonzero(values < np.roll(values, -1))


def compute_next_ridge_following_directions(previous_direction, values):
    next_positions = np.argwhere(values != 0).ravel().tolist()
    if len(next_positions) > 0 and previous_direction != 8:
        # There is a previous direction: return all the next directions, sorted according to the distance from it,
        # except the direction, if any, that corresponds to the previous position
        next_positions.sort(key=lambda d: 4 - abs(abs(d - previous_direction) - 4))
        if next_positions[-1] == (
                previous_direction + 4) % 8:  # the direction of the previous position is the opposite one
            next_positions = next_positions[:-1]  # removes it
    return next_positions


def follow_ridge_and_compute_angle(x, y, nd_lut, cn_values, cn, xy_steps, d=8):
    px, py = x, y
    length = 0.0
    while length < 20:  # max length followed
        next_directions = nd_lut[cn_values[py, px]][d]
        if len(next_directions) == 0:
            break
        # Need to check ALL possible next directions
        if (any(cn[py + xy_steps[nd][1], px + xy_steps[nd][0]] != 2 for nd in next_directions)):
            break  # another minutia found: we stop here
        # Only the first direction has to be followed
        d = next_directions[0]
        ox, oy, l = xy_steps[d]
        px += ox;
        py += oy;
        length += l
    # check if the minimum length for a valid direction has been reached
    return math.atan2(-py + y, px - x) if length >= 10 else None


def imageProcess(img):
    fingerprint = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    # print(fingerprint)
    # cv2.imshow(f'Fingerprint with size (w,h): {fingerprint.shape[::-1]}',fingerprint)
    gx, gy = cv2.Sobel(fingerprint, cv2.CV_32F, 1, 0), cv2.Sobel(fingerprint, cv2.CV_32F, 0, 1)
    gx2, gy2 = gx ** 2, gy ** 2
    gm = np.sqrt(gx2 + gy2)
    sum_gm = cv2.boxFilter(gm, -1, (25, 25), normalize=False)
    thr = sum_gm.max() * 0.2
    mask = cv2.threshold(sum_gm, thr, 255, cv2.THRESH_BINARY)[1].astype(np.uint8)
    W = (23, 23)  # 我们定义一个23x23的窗口
    gxx = cv2.boxFilter(gx2, -1, W, normalize=False)  # 在给定的滑动窗口大小下，对每个窗口内的像素值进行快速相加求和
    gyy = cv2.boxFilter(gy2, -1, W, normalize=False)
    gxy = cv2.boxFilter(gx * gy, -1, W, normalize=False)  # gx * gy
    gxx_gyy = gxx - gyy
    gxy2 = 2 * gxy
    orientations = (cv2.phase(gxx_gyy, -gxy2) + np.pi) / 2  # '-' to adjust for y axis direction phase函数计算方向场
    sum_gxx_gyy = gxx + gyy
    strengths = np.divide(cv2.sqrt((gxx_gyy ** 2 + gxy2 ** 2)), sum_gxx_gyy, out=np.zeros_like(gxx),
                          where=sum_gxx_gyy != 0)  # 计算置信度，也就是计算在W 中有多少梯度有同样的方向，自然数量越多，计算的结果越可靠
    region = fingerprint[10:90, 80:130]
    smoothed = cv2.blur(region, (5, 5), -1)
    xs = np.sum(smoothed, 1)  # the x-signature of the region
    # Find the indices of the x-signature local maxima
    local_maxima = np.nonzero(np.r_[False, xs[1:] > xs[:-1]] & np.r_[xs[:-1] >= xs[1:], False])[0]
    distances = local_maxima[1:] - local_maxima[:-1]
    ridge_period = np.average(distances)
    # print(ridge_period)
    or_count = 8
    gabor_bank = [gabor_kernel(ridge_period, o) for o in np.arange(0, np.pi, np.pi / or_count)]
    nf = 255 - fingerprint
    all_filtered = np.array([cv2.filter2D(nf, cv2.CV_32F, f) for f in gabor_bank])
    # print(type(gabor_bank[0]))
    y_coords, x_coords = np.indices(fingerprint.shape)
    # For each pixel, find the index of the closest orientation in the gabor bank
    orientation_idx = np.round(((orientations % np.pi) / np.pi) * or_count).astype(np.int32) % or_count
    # Take the corresponding convolution result for each pixel, to assemble the final result
    filtered = all_filtered[orientation_idx, y_coords, x_coords]
    # Convert to gray scale and apply the mask
    enhanced = mask & np.clip(filtered, 0, 255).astype(np.uint8)
    # Binarization
    _, ridge_lines = cv2.threshold(enhanced, 32, 255, cv2.THRESH_BINARY)  # enhanced 是增强之后的图像
    skeleton = cv2.ximgproc.thinning(ridge_lines, thinningType=cv2.ximgproc.THINNING_GUOHALL)
    # Create a filter that converts any 8-neighborhood into the corresponding byte value [0,255]
    cn_filter = np.array([[1, 2, 4],
                          [128, 0, 8],
                          [64, 32, 16]
                          ])
    # Create a lookup table that maps each byte value to the corresponding crossing number
    all_8_neighborhoods = [np.array([int(d) for d in f'{x:08b}'])[::-1] for x in range(256)]
    cn_lut = np.array([compute_crossing_number(x) for x in all_8_neighborhoods]).astype(np.uint8)
    # Skeleton: from 0/255 to 0/1 values
    skeleton01 = np.where(skeleton != 0, 1, 0).astype(np.uint8)
    # 应用设计好的filter，将8领域转换为0-255的byte
    cn_values = cv2.filter2D(skeleton01, -1, cn_filter, borderType=cv2.BORDER_CONSTANT)
    # 使用查找表，获取crossing numbers的值
    cn = cv2.LUT(cn_values, cn_lut)
    # Keep only crossing numbers on the skeleton
    cn[skeleton == 0] = 0
    # crossing number == 1 --> Termination, crossing number == 3 --> Bifurcation
    minutiae = [(x, y, cn[y, x] == 1) for y, x in zip(*np.where(np.isin(cn, [1, 3])))]
    # A 1-pixel background border is added to the mask before computing the distance transform
    mask_distance = cv2.distanceTransform(cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT), cv2.DIST_C, 3)[
                    1:-1, 1:-1]
    filtered_minutiae = list(filter(lambda m: mask_distance[m[1], m[0]] > 10, minutiae))
    # print(filtered_minutiae)

    r2 = 2 ** 0.5  # sqrt(2)

    # The eight possible (x, y) offsets with each corresponding Euclidean distance
    xy_steps = [(-1, -1, r2), (0, -1, 1), (1, -1, r2), (1, 0, 1), (1, 1, r2), (0, 1, 1), (-1, 1, r2), (-1, 0, 1)]

    # LUT: for each 8-neighborhood and each previous direction [0,8],
    #      where 8 means "none", provides the list of possible directions
    nd_lut = [[compute_next_ridge_following_directions(pd, x) for pd in range(9)] for x in all_8_neighborhoods]

    valid_minutiae = []
    for x, y, term in filtered_minutiae:
        d = None
        if term:  # termination: simply follow and compute the direction
            d = follow_ridge_and_compute_angle(x, y, d=8, nd_lut=nd_lut, cn_values=cn_values, cn=cn, xy_steps=xy_steps)
        else:  # bifurcation: follow each of the three branches
            dirs = nd_lut[cn_values[y, x]][8]  # 8 means: no previous direction
            if len(dirs) == 3:  # only if there are exactly three branches
                angles = [follow_ridge_and_compute_angle(x + xy_steps[d][0], y + xy_steps[d][1], d=d, nd_lut=nd_lut,
                                                         cn_values=cn_values, cn=cn, xy_steps=xy_steps) for d in dirs]
                if all(a is not None for a in angles):
                    a1, a2 = min(((angles[i], angles[(i + 1) % 3]) for i in range(3)),
                                 key=lambda t: abs(t[0] - t[1]))
                    d = (a1 + a2) / 2
        if d is not None:
            valid_minutiae.append((x, y, term, d))

    # print(valid_minutiae)
    # Compute the cell coordinates of a generic local structure
    # 计算
    mcc_radius = 70
    mcc_size = 16

    g = 2 * mcc_radius / mcc_size
    x = np.arange(mcc_size) * g - (mcc_size / 2) * g + g / 2
    y = x[..., np.newaxis]
    iy, ix = np.nonzero(x ** 2 + y ** 2 <= mcc_radius ** 2)
    ref_cell_coords = np.column_stack((x[ix], x[iy]))
    # n: number of minutiae
    # c: number of cells in a local structure

    xyd = np.array(
        [(x, y, d) for x, y, _, d in valid_minutiae])  # matrix with all minutiae coordinates and directions (n x 3)

    # rot: n x 2 x 2 (rotation matrix for each minutia)
    d_cos, d_sin = np.cos(xyd[:, 2]).reshape((-1, 1, 1)), np.sin(xyd[:, 2]).reshape((-1, 1, 1))
    rot = np.block([[d_cos, d_sin], [-d_sin, d_cos]])

    # rot@ref_cell_coords.T : n x 2 x c
    # xy : n x 2
    xy = xyd[:, :2]
    # cell_coords: n x c x 2 (cell coordinates for each local structure)
    cell_coords = np.transpose(rot @ ref_cell_coords.T + xy[:, :, np.newaxis], [0, 2, 1])

    # cell_coords[:,:,np.newaxis,:]      :  n x c  x 1 x 2
    # xy                                 : (1 x 1) x n x 2
    # cell_coords[:,:,np.newaxis,:] - xy :  n x c  x n x 2
    # dists: n x c x n (for each cell of each local structure, the distance from all minutiae)
    dists = np.sum((cell_coords[:, :, np.newaxis, :] - xy) ** 2, -1)

    # cs : n x c x n (the spatial contribution of each minutia to each cell of each local structure)
    cs = Gs(dists)
    diag_indices = np.arange(cs.shape[0])
    cs[diag_indices, :, diag_indices] = 0  # remove the contribution of each minutia to its own cells

    # local_structures : n x c (cell values for each local structure)
    local_structures = Psi(np.sum(cs, -1))
    print(f"""Fingerprint image: {fingerprint.shape[1]}x{fingerprint.shape[0]} pixels
    Minutiae: {len(valid_minutiae)}
    Local structures: {local_structures.shape}""")
    f1, m1, ls1 = fingerprint, valid_minutiae, local_structures
    return f1, m1, ls1


def getScore(f1, m1, ls1, f2, m2, ls2, confidence=0.7):
    dists = np.sqrt(np.sum((ls1[:, np.newaxis, :] - ls2) ** 2, -1))
    dists /= (np.sqrt(np.sum(ls1 ** 2, 1))[:, np.newaxis] + np.sqrt(
        np.sum(ls2 ** 2, 1)))  # Normalize as in eq. (17) of MCC paper
    # Select the num_p pairs with the smallest distances (LSS technique)
    num_p = 5  # For simplicity: a fixed number of pairs
    pairs = np.unravel_index(np.argpartition(dists, num_p, None)[:num_p], dists.shape)
    score = 1 - np.mean(dists[pairs[0], pairs[1]])  # See eq. (23) in MCC paper
    print(f'Comparison score: {score:.2f}')  # 结果0.78
    if score > confidence:
        print('匹配')
        return score, 1
    else:
        print('不匹配')
        return score, 0


def Comparison(img1, img2, confidence=0.7):
    f1, m1, ls1 = imageProcess(img1)
    f2, m2, ls2 = imageProcess(img2)
    return getScore(f1, m1, ls1, f2, m2, ls2, confidence)
