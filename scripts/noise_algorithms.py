import numpy as np
from scipy.spatial import Voronoi
import colorsys
import cv2


def generate_simplex_noise(width, height, scale, octaves, persistence, seed):
    def simplex2d(x, y):
        def fast_floor(x):
            return int(x) if x > 0 else int(x) - 1

        def dot(g, x, y):
            return g[0] * x + g[1] * y

        F2 = 0.5 * (np.sqrt(3.0) - 1.0)
        G2 = (3.0 - np.sqrt(3.0)) / 6.0

        s = (x + y) * F2
        i = fast_floor(x + s)
        j = fast_floor(y + s)
        t = (i + j) * G2

        X0 = i - t
        Y0 = j - t
        x0 = x - X0
        y0 = y - Y0

        if x0 > y0:
            i1, j1 = 1, 0
        else:
            i1, j1 = 0, 1

        x1 = x0 - i1 + G2
        y1 = y0 - j1 + G2
        x2 = x0 - 1.0 + 2.0 * G2
        y2 = y0 - 1.0 + 2.0 * G2

        ii = i & 255
        jj = j & 255

        gi0 = perm[ii + perm[jj]] % 12
        gi1 = perm[ii + i1 + perm[jj + j1]] % 12
        gi2 = perm[ii + 1 + perm[jj + 1]] % 12

        t0 = 0.5 - x0 * x0 - y0 * y0
        if t0 < 0:
            n0 = 0.0
        else:
            t0 *= t0
            n0 = t0 * t0 * dot(grad3[gi0], x0, y0)

        t1 = 0.5 - x1 * x1 - y1 * y1
        if t1 < 0:
            n1 = 0.0
        else:
            t1 *= t1
            n1 = t1 * t1 * dot(grad3[gi1], x1, y1)

        t2 = 0.5 - x2 * x2 - y2 * y2
        if t2 < 0:
            n2 = 0.0
        else:
            t2 *= t2
            n2 = t2 * t2 * dot(grad3[gi2], x2, y2)

        return 70.0 * (n0 + n1 + n2)

    np.random.seed(seed)
    perm = list(range(256))
    np.random.shuffle(perm)
    perm += perm
    grad3 = [(1,1,0),(-1,1,0),(1,-1,0),(-1,-1,0),
             (1,0,1),(-1,0,1),(1,0,-1),(-1,0,-1),
             (0,1,1),(0,-1,1),(0,1,-1),(0,-1,-1)]

    scale /= 100.0
    noise_array = np.zeros((height, width))
    for octave in range(octaves):
        frequency = 2 ** octave
        amplitude = persistence ** octave
        for y in range(height):
            for x in range(width):
                noise_value = simplex2d(x * scale * frequency, y * scale * frequency)
                noise_array[y, x] += noise_value * amplitude

    noise_array = (noise_array - noise_array.min()) / (noise_array.max() - noise_array.min())
    return noise_array

def generate_marble_noise(width, height, scale, octaves, persistence, seed):
    simplex_noise = generate_simplex_noise(width, height, scale, octaves, persistence, seed)
    x = np.linspace(0, 10, width)
    y = np.linspace(0, 10, height)
    x, y = np.meshgrid(x, y)
    
    marble = np.sin(x + 10 * simplex_noise)
    marble = (marble - marble.min()) / (marble.max() - marble.min())
    return marble

def generate_water_splash_noise(width, height, scale, octaves, persistence, seed):
    simplex_noise = generate_simplex_noise(width, height, scale, octaves, persistence, seed)
    
    np.random.seed(seed)
    num_drops = 20
    centers = np.random.rand(num_drops, 2) * np.array([width, height])
    
    splash = np.zeros((height, width))
    for center in centers:
        x = np.arange(width)
        y = np.arange(height)
        x, y = np.meshgrid(x, y)
        distance = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        splash += np.exp(-distance / (width/10)) * simplex_noise
    
    splash = (splash - splash.min()) / (splash.max() - splash.min())
    return splash

def generate_colorful_voronoi(width, height, num_cells, seed):
    np.random.seed(seed)
    
    # ランダムな点を生成
    points = np.random.rand(num_cells, 2)
    points[:, 0] *= width
    points[:, 1] *= height
    
    # ボロノイ図を生成
    vor = Voronoi(points)
    
    # 色を生成
    colors = [tuple(int(x * 255) for x in colorsys.hsv_to_rgb(np.random.rand(), 0.8, 1.0)) for _ in range(len(vor.points))]
    
    # 画像を生成
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 各領域を塗りつぶす
    for i, region in enumerate(vor.regions):
        if not -1 in region:
            polygon = [vor.vertices[v] for v in region]
            polygon = np.array(polygon, dtype=np.int32)
            cv2.fillPoly(image, [polygon], colors[i % len(colors)])
    
    # 境界を処理
    center = vor.points.mean(axis=0)
    for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        simplex = np.asarray(simplex)
        if -1 in simplex:
            i = simplex[simplex >= 0][0]  # 有効な頂点のインデックス
            t = vor.points[pointidx[1]] - vor.points[pointidx[0]]  # 接線
            t = t / np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # 法線
            midpoint = vor.points[pointidx].mean(axis=0)
            far_point = vor.vertices[i] + np.sign(np.dot(midpoint - center, n)) * n * max(width, height)
            pts = np.vstack((vor.vertices[i], far_point)).astype(int)
            cv2.fillPoly(image, [pts], colors[pointidx[0]])

    return image