#!/usr/bin/env python3
"""extract_dims_robust.py

Amélioration du script d'extraction de dimensions d'un nuage de points
cuboïdal partiel. Fonctionne même lorsqu'une seule ou deux faces sont
visibles (p.ex. top + une face latérale, ou deux faces latérales).

Ajout de visualisations contrôlées pour le nuage nettoyé, les plans détectés et la boîte englobante.

Convention : la dimension horizontale la plus grande est toujours appelée
« length », la plus petite « width », quelle que soit leur orientation.

Usage :
    python extract_dims_robust.py <cloud.[ply|pcd]> [--no-visualize] [--voxel 0.003]

Nécessite : numpy, open3d >= 0.18
"""

from __future__ import annotations
import argparse
from typing import List, Tuple

import numpy as np
import open3d as o3d

# --------------------------------------------------------------------------- #
#  Pré‑traitements                                                            #
# --------------------------------------------------------------------------- #

def statistical_clean(pcd: o3d.geometry.PointCloud,
                      nb_neighbors: int = 30,
                      std_ratio: float = 2.0) -> o3d.geometry.PointCloud:
    """Supprime les outliers statistiques."""
    cl, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return cl

def downsample(pcd: o3d.geometry.PointCloud, voxel: float | None) -> o3d.geometry.PointCloud:
    """Échantillonnage voxel facultatif pour accélérer le traitement."""
    return pcd.voxel_down_sample(voxel) if voxel and voxel > 0 else pcd

# --------------------------------------------------------------------------- #
#  Segmentation des plans                                                     #
# --------------------------------------------------------------------------- #

def segment_planes(pcd: o3d.geometry.PointCloud,
                   distance_threshold: float = 0.005,
                   ransac_n: int = 3,
                   num_iterations: int = 1_000,
                   max_planes: int = 5,
                   min_inliers: int = 200,
                   normal_sim_thresh: float = 0.95) -> Tuple[List[np.ndarray], List[o3d.geometry.PointCloud]]:
    """Renvoie un ensemble de normales de plans et les nuages de points inliers."""
    normals: List[np.ndarray] = []
    inlier_clouds: List[o3d.geometry.PointCloud] = []
    rest = pcd
    for _ in range(max_planes):
        if len(rest.points) < 50:
            break
        model, inliers = rest.segment_plane(distance_threshold=distance_threshold,
                                            ransac_n=ransac_n,
                                            num_iterations=num_iterations)
        if len(inliers) < min_inliers:
            break
        n = np.asarray(model[:3], dtype=float)
        n /= np.linalg.norm(n)
        if all(abs(float(n @ m)) < normal_sim_thresh for m in normals):
            normals.append(n)
            inlier_cloud = rest.select_by_index(inliers)
            inlier_clouds.append(inlier_cloud)
        rest = rest.select_by_index(inliers, invert=True)
        if len(normals) >= 3:
            break
    return normals, inlier_clouds

# --------------------------------------------------------------------------- #
#  Complément d'axes et orthonormalisation                                    #
# --------------------------------------------------------------------------- #

def pca_axes(pcd: o3d.geometry.PointCloud) -> List[np.ndarray]:
    pts = np.asarray(pcd.points, dtype=float)
    cov = np.cov(pts.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    return [eigvecs[:, i] / np.linalg.norm(eigvecs[:, i]) for i in order]

def complete_axes(normals: List[np.ndarray],
                  pcd: o3d.geometry.PointCloud,
                  sim_thresh: float = 0.95) -> List[np.ndarray]:
    axes = list(normals)
    for a in pca_axes(pcd):
        if all(abs(float(a @ b)) < sim_thresh for b in axes):
            axes.append(a)
        if len(axes) == 3:
            break
    return axes

def gram_schmidt(axes: List[np.ndarray]) -> List[np.ndarray]:
    u0 = axes[0] / np.linalg.norm(axes[0])
    u1 = axes[1] - (axes[1] @ u0) * u0
    u1 /= np.linalg.norm(u1)
    u2 = np.cross(u0, u1)
    if axes[2] @ u2 < 0:
        u2 = -u2
    return [u0, u1, u2 / np.linalg.norm(u2)]

# --------------------------------------------------------------------------- #
#  Ordonnancement : Height | Width | Length                                   #
# --------------------------------------------------------------------------- #

def order_axes(axes: List[np.ndarray]) -> Tuple[List[np.ndarray], List[str]]:
    """Retourne les axes dans l’ordre (height, width, length)."""
    z = np.array([0, 0, 1], dtype=float)
    dots = [abs(float(a @ z)) for a in axes]
    h_idx = int(np.argmax(dots))
    order = [h_idx, (h_idx + 1) % 3, (h_idx + 2) % 3]
    names = ["height", "width", "length"]
    return [axes[i] for i in order], [names[i] for i in order]

# --------------------------------------------------------------------------- #
#  Étendue robuste d’un axe                                                   #
# --------------------------------------------------------------------------- #

def robust_extent(pts: np.ndarray,
                  axis: np.ndarray,
                  low_q: float = 0.01,
                  high_q: float = 0.99) -> float:
    proj = pts @ axis
    return float(np.quantile(proj, high_q) - np.quantile(proj, low_q))

# --------------------------------------------------------------------------- #
#  Pipeline principal                                                         #
# --------------------------------------------------------------------------- #

def extract_dimensions(pcd: o3d.geometry.PointCloud,
                       voxel: float | None = None,
                       plane_th: float = 0.005,
                       visualize: bool = True) -> Tuple[dict[str, float], List[np.ndarray]]:
    # Nettoyage et échantillonnage
    p_work = statistical_clean(downsample(pcd, voxel))
    if visualize:
        print("Visualisation du nuage nettoyé et échantillonné...")
        p_work_copy = o3d.geometry.PointCloud(p_work)  # Copie pour éviter de modifier
        p_work_copy.paint_uniform_color([0.6, 0.6, 0.6])
        o3d.visualization.draw_geometries([p_work_copy], window_name="Nuage nettoyé", width=900, height=600)

    # Détection des plans
    normals, inlier_clouds = segment_planes(p_work, distance_threshold=plane_th)
    if visualize and inlier_clouds:
        print("Visualisation des plans détectés...")
        colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1]]  # Rouge, vert, bleu, jaune, magenta
        for i, cloud in enumerate(inlier_clouds):
            cloud.paint_uniform_color(colors[i % len(colors)])
        o3d.visualization.draw_geometries(
            inlier_clouds,
            window_name="Plans détectés (inliers)",
            width=900,
            height=600
        )

    # Calcul des axes et dimensions
    axes = gram_schmidt(complete_axes(normals, p_work))
    axes, names = order_axes(axes)
    pts = np.asarray(p_work.points, dtype=float)
    dims = [robust_extent(pts, a) for a in axes]
    dims_named = dict(zip(names, dims))

    # Convention : length >= width
    if "width" in dims_named and "length" in dims_named and dims_named["width"] > dims_named["length"]:
        dims_named["width"], dims_named["length"] = dims_named["length"], dims_named["width"]
        axes[1], axes[2] = axes[2], axes[1]

    return dims_named, axes

# --------------------------------------------------------------------------- #
#  OBB de visualisation                                                       #
# --------------------------------------------------------------------------- #

def build_obb(center: np.ndarray,
              axes: List[np.ndarray],
              dims_named: dict[str, float]) -> o3d.geometry.OrientedBoundingBox:
    dims = [dims_named.get("width", 0), dims_named.get("length", 0), dims_named.get("height", 0)]
    R = np.stack(axes, axis=1)
    return o3d.geometry.OrientedBoundingBox(center=center, R=R, extent=dims)

# --------------------------------------------------------------------------- #
#  Entrée / Sortie CLI                                                        #
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description="Extraction robuste des dimensions W×L×H d’un nuage de cuboïde partiel")
    parser.add_argument("filename", help="fichier PLY/PCD du nuage")
    parser.add_argument("--no-visualize", action="store_true", help="désactive les visualisations intermédiaires")
    parser.add_argument("--voxel", type=float, metavar="SIZE", help="taille voxel pour downsample (ex : 0.003)")
    parser.add_argument("--plane_th", type=float, default=0.005, help="seuil RANSAC plan (m, défaut : 0.005)")
    args = parser.parse_args()

    pcd = o3d.io.read_point_cloud(args.filename)
    if not pcd.has_points():
        raise RuntimeError(f"Nuage vide ou fichier introuvable : {args.filename}")

    dims_named, axes = extract_dimensions(pcd, voxel=args.voxel, plane_th=args.plane_th, visualize=not args.no_visualize)

    print("Dimensions estimées (m) :")
    for k in ("width", "length", "height"):
        print(f"  {k.capitalize():7}: {dims_named.get(k, float('nan')):.4f}")

    # Visualisation de la boîte englobante (toujours affichée)
    print("Visualisation de la boîte englobante...")
    obb = build_obb(np.mean(np.asarray(pcd.points), axis=0), axes, dims_named)
    obb.color = (1.0, 0.0, 0.0)
    pcd_copy = o3d.geometry.PointCloud(pcd)  # Copie pour éviter de modifier
    pcd_copy.paint_uniform_color([0.6, 0.6, 0.6])
    o3d.visualization.draw_geometries([pcd_copy, obb], window_name="Boîte englobante ajustée", width=900, height=600)

if __name__ == "__main__":
    main()