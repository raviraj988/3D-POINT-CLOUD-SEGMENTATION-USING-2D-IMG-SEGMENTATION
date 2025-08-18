from pathlib import Path
import time
import os

import numpy as np
import open3d as o3d

from Fusion3DSeg.fusion import Fusion





def process3DSeg(
        input_data_path, output_path,
        radius=0.05, angle=10, stride=10,
        point_range=(0.1, 4), decimation=1, min_occ=3,
        verbose=False,
    ):
    # ============================================================
    # Cameradata path extraction
    # ============================================================
    mergeresults_path = os.path.join(input_data_path, 'PointcloudMergeResults')
    if os.path.exists(mergeresults_path):
        ss = [f for f in os.listdir(input_data_path + os.sep + 'PointcloudMergeResults') if f.__contains__('tofsegment')][0][:-4]
        subfilename = ss.split('_', 1)[1]
    else:
        print('tofcameradata not found')

    tof = os.path.join(mergeresults_path, f"tofsegment_{subfilename}.pkl")
    rts = os.path.join(mergeresults_path, f"rtscameradata_{subfilename}.pkl")

    # =============================================================
    # Point Cloud Fusion + Down Sampling
    # =============================================================
    start = time.perf_counter()
    fuser = Fusion(tof, rts, point_range, decimation)
    ds_pts, ds_norms, ds_clrs, nmerges, occurences = fuser.fuse(
        radius, angle, stride, point_range[1], skip=1, verbose=verbose
    )
    end = time.perf_counter()

    if verbose:
        print(f'\ntotal {fuser.npts*fuser.nframes} points from {fuser.nframes} frames are fused into {len(ds_pts)} points')
        print(f'time taken for fusion = {(end - start)/60} minutes')

    # =============================================================
    # Occurence Denoising
    # =============================================================
    if min_occ is not None:
        mask, (ds_pts_, ds_norms_, ds_clrs_, nmerges_, occurences_) = fuser.filter(
            nmerges, min_occ, [ds_pts, ds_norms, ds_clrs, nmerges, occurences], less_than=False,
        )

        if verbose: print(f'remaining points after frame occurence thresholding with {min_occ} = {mask.sum()}')

    # =============================================================
    # Visualization + Data Dump
    # =============================================================
    # dirname = Path(rts.split('PointcloudMergeResults')[0])
    dirname = Path(output_path)
    radius = str(radius).replace('.', '_')
    fuser.dump_data(dirname, ds_pts, ds_norms, ds_clrs, nmerges, occurences, True, verbose)
    ds_pts, ds_norms, ds_clrs, nmerges, occurences, nframes, hw, adj = fuser.load_data(dirname)
    # =============================================================
    # End
    # =============================================================
    return ds_pts, ds_norms, ds_clrs, nmerges, occurences, nframes, hw, adj


if __name__ == '__main__':
    # =============================================================
    # Parameters
    # =============================================================
    input_data_path = 'test_data/rtab'

    process3DSeg(
        input_data_path, 
        input_data_path,
        radius = 0.05, 
        angle = 10,
        stride = 10,
        point_range = (0.1, 4),
        decimation = 1,
        min_occ = 3,
        verbose = False,
    )

