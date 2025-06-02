import argparse, sys, numpy as np, open3d as o3d

def isolate_auto(pcd, voxel, plane_d, eps, min_pts, keep_normals):
    # ↓ 1. (facultatif) on garde les normales si demandé
    if keep_normals:
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(30))

    # 2. Voxel down-sampling
    if voxel > 0:
        pcd = pcd.voxel_down_sample(voxel)
    
    # 3. Outliers
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    # 4. Plan (sol/table) – on force la normale quasi verticale si besoin
    plane_model, inliers = pcd.segment_plane(plane_d, 3, 1000)
    pcd = pcd.select_by_index(inliers, invert=True)

    # 5. Clusterisation
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_pts))
    valid  = labels[labels >= 0]
    if valid.size == 0:
        raise RuntimeError("Aucun cluster détecté ; réduis min_points ou augmente eps.")
    largest = np.bincount(valid).argmax()
    return pcd.select_by_index(np.where(labels == largest)[0])

# ------------------------------------------------------------------
# Lancement : python isoler_ply_tunable.py scene.ply --voxel 0.001 --min_pts 10
# ------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("ply", help="fichier .ply source")
parser.add_argument("-o", "--out", default="objet_seul.ply")
parser.add_argument("--voxel",    type=float, default=0.001, help="0=pas de down-sampling (m)")
parser.add_argument("--plane_d",  type=float, default=0.004, help="tolérance plan RANSAC (m)")
parser.add_argument("--eps",      type=float, default=0.015, help="rayon DBSCAN (m)")
parser.add_argument("--min_pts",  type=int,   default=15,    help="min_pts DBSCAN")
parser.add_argument("--keep_normals", action="store_true", help="calcule et garde les normales")
args = parser.parse_args()

pcd = o3d.io.read_point_cloud(args.ply)
try:
    obj = isolate_auto(pcd, args.voxel, args.plane_d, args.eps, args.min_pts,
                       args.keep_normals)
except Exception as e:
    sys.exit(f"❌  {e}")

o3d.io.write_point_cloud(args.out, obj)
print(f"✅  {args.out}  ({len(obj.points)} pts)")
o3d.visualization.draw_geometries([obj], window_name="Objet isolé")
