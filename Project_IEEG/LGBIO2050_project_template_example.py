"""
[LGBIO2050] - Medical Imaging
Project: Localizing iEEG contacts in a CT scan - code template

Professors: Prof. G. Kerckhofs, Prof. J. Lee, Prof. B. Macq, Prof. G. Duchêne
Teaching assistants:
    Pauline Hermans (pauline.hermans@uclouvain.be)
    Nicolas Delinte (nicolas.delinte@uclouvain.be)
    Simon Francois (simon.b.francois@uclouvain.be)
2026-2027

Run this script from its own folder. It writes placeholder outputs until you
replace the TODOs.
"""

import os
import numpy as np
import nibabel as nib
import pyvista as pv
import matplotlib.pyplot as plt

import scipy
import skimage
from tqdm import tqdm

# Tools and functions (feel free to add your own)


def load_nifti(path):
    """
    Load a .nii or .nii.gz file. Returns the volume, read as volume[i, j, k],
    and the size of a voxel along i, j and k, in mm.

    Parameters
    ----------
    path : str

    Returns
    -------
    volume : numpy array
    """
    image = nib.load(path)
    # float32 uses half the memory of float64
    volume = image.get_fdata(dtype=np.float32)

    return volume


def save_nifti(volume, output_path, ct_path, dtype=np.uint8):
    """
    Save a volume as a NIfTI file.

    The affine information is retrieve from ct_path so that the segmentation
    overlay correctly the CT.

    The default dtype suits masks and segmentations. Use dtype=np.float32
    for other volumes.

    Parameters
    ----------
    volume : numpy array
    output_path : str
    ct_path : str
    dtype : TYPE, optional
        The default is np.uint8.

    Returns
    -------
    None
    """
    affine = nib.load(ct_path).affine
    nii_img = nib.Nifti1Image(np.asarray(volume).astype(dtype), affine)
    nib.save(nii_img, output_path)


def interactive_plot(CT_filepath, segmentation_filepath=None, shown_masks=None,
                     points_filepath=None):
    """
    Open an interactive 3D view of a CT volume, with optional surfaces from
    a segmentation file (NIfTI) and optional contact points (.npy).

    Parameters
    ----------
    CT_filepath : str
        Path to the CT NIfTI file.
    segmentation_filepath : str, optional
        Path to a segmentation NIfTI file on the same grid as the CT.
        One smoothed surface is drawn per non-zero masks (0 = background).
        If None, no surfaces are drawn.
    shown_masks : int or list of int, optional
        Mask(s) of the segmentation file to draw. If None, all non-zero
        masks are drawn. Masks absent from the file are ignored.
        Unused without a segmentation file.
    points_filepath : str, optional
        Path to a .npy file of contact coordinates in voxel space.
        Shape (n, 3): x, y, z -> all points in red.
        Shape (n, 4): x, y, z, label -> one color per electrode label.
    """
    CT = nib.load(CT_filepath).get_fdata(dtype=np.float32)
    grid = pv.ImageData(dimensions=np.array(CT.shape) + 1)
    grid.cell_data['CT'] = CT.ravel(order='F')
    del CT

    plotter = pv.Plotter()
    plotter.add_volume(grid, scalars='CT', cmap='gray', opacity=[0.0, 0.04],
                       show_scalar_bar=False)
    legend = False

    # One smoothed surface per segment of the segmentation mask
    if segmentation_filepath is not None:
        seg = nib.load(segmentation_filepath).get_fdata().astype(int)
        grid.cell_data['segmentation'] = seg.ravel(order='F')

        chosen = np.unique(seg[seg != 0])
        if shown_masks is not None:
            chosen = np.intersect1d(np.atleast_1d(shown_masks), chosen)

        colors = plt.get_cmap('Paired', max(len(chosen)+1, 1))
        for i, s in enumerate(chosen):
            surface = grid.threshold(
                [s, s], scalars='segmentation').extract_surface(
                algorithm='dataset_surface')
            plotter.add_mesh(surface.smooth_taubin(n_iter=12),
                             color=colors(i)[:3], label=f'Mask {s}')
            legend = True

    # Contact points, one color per electrode label if a 4th column is given
    if points_filepath is not None:
        points = np.load(points_filepath)
        point_labels = points[:, 3] if points.shape[1] == 4 else np.zeros(
            len(points))
        unique_labels = np.unique(point_labels)
        colors = plt.get_cmap('gist_rainbow', len(unique_labels))

        for i, lab in enumerate(unique_labels):
            color = 'red' if points.shape[1] == 3 else colors(i)[:3]
            name = 'Contacts' if points.shape[1] == 3 else f'Electrode {lab:g}'
            plotter.add_mesh(pv.PolyData(points[point_labels == lab, :3].astype(float)),
                             color=color, label=name, point_size=10,
                             render_points_as_spheres=True)
            legend = True

    if legend:
        plotter.add_legend()

    plotter.show()
    return


if __name__ == '__main__':

    # %% Set paths and load CT

    sub = "sub-01"
    ct_path = f"data/{sub}_CT.nii.gz"

    os.makedirs("output", exist_ok=True)

    ct = load_nifti(ct_path)
    print("CT shape:", ct.shape)

    # %% Task A: preliminary segmentation

    # TODO
    segmentation = np.zeros(ct.shape, dtype=np.uint8)

    task1_path = f"output/{sub}_task-1_segmentation.nii.gz"
    save_nifti(segmentation, task1_path, ct_path)

    # %% Task B: brain mask

    # TODO
    brain_mask = np.zeros(ct.shape, dtype=np.uint8)

    task2_path = f"output/{sub}_task-2_brain-mask.nii.gz"
    save_nifti(brain_mask, task2_path, ct_path)

    # %% Task C: coordinates of the electrode contacts

    # TODO

    some_coordinates = np.array([[250, 200, 100, 0],
                                 [260, 200, 100, 0],
                                 [270, 200, 100, 0],
                                 [280, 200, 100, 0],
                                 [290, 200, 100, 0],
                                 [200, 100, 110, 1],
                                 [200, 100, 120, 1],
                                 [200, 100, 130, 1],
                                 [200, 100, 140, 1],
                                 [200, 100, 150, 1]])

    task3_path = f"output/{sub}_task-3_coordinates.npy"
    np.save(task3_path, some_coordinates)

    # %% Visualization function

    # The 3 first plots will only show the CT as the segmantion files are
    # currently all zeros. Complete the TODOs to see them on the plot.
    # The plots are not interactive in Jupyter Notebook.
    # Adjust the function arguments to control what is displayed.

    # Show soft-tissue mask
    interactive_plot(
        ct_path, segmentation_filepath=task1_path, shown_masks=[1])

    # Show bone and metal mask
    interactive_plot(ct_path, segmentation_filepath=task1_path,
                     shown_masks=[2, 3])

    # Show brain mask
    interactive_plot(ct_path, segmentation_filepath=task2_path)

    # Show contacts points
    interactive_plot(ct_path, points_filepath=task3_path)
