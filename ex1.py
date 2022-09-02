#207237421 Yael Avioz
import matplotlib.pyplot as plt
import numpy as np
import sys


def is_same_centers(old_centers, z):
    for i in range(len(z)):
        if (old_centers[i] != z[i]).any():
            return False
    return True


def find_nearest_center(pixel, z):
    is_first_iter = True
    for center_idx, center in enumerate(z):
        # calc the distance from the center
        dist = np.linalg.norm(pixel - center)
        if is_first_iter:
            cur_center_idx = center_idx
            min_dist = dist
            is_first_iter = False
        elif dist < min_dist:
            cur_center_idx = center_idx
            min_dist = dist
    return cur_center_idx, min_dist


if __name__ == "__main__":
    image_fname, centroids_fname, out_fname = sys.argv[1], sys.argv[2], sys.argv[3]
    z = np.loadtxt(centroids_fname)  # load centroids

    orig_pixels = plt.imread(image_fname)
    pixels = orig_pixels.astype(float) / 255.
    # Reshape the image(128x128x3) into an Nx3 matrix where N = number of pixels.
    pixels = pixels.reshape(-1, 3)

    k = len(z)
    centers_lists = [[] for i in range(k)]
    iter_num = 0
    new_centers = z
    old_centers = np.copy(new_centers)
    is_first_iter = True
    with open(out_fname, "w") as f:
        while (is_first_iter) or ((not is_same_centers(old_centers, new_centers)) and (iter_num < 20)):
            is_first_iter = False
            centers_lists = [[] for i in range(k)]
            for pixel in pixels:
                nearest_center_idx, dist = find_nearest_center(pixel, z)
                centers_lists[nearest_center_idx].append(pixel)

            old_centers = np.copy(new_centers)

            for i, center_list in enumerate(centers_lists):

                if len(center_list) > 0:
                    new_centers[i] = np.array(center_list).mean(axis=0).round(4)
            f.write("[iter {}]:{}\n".format(iter_num, ','.join([str(i) for i in new_centers])))
            iter_num += 1
