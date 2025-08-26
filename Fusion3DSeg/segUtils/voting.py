from pathlib import Path

import numpy as np
import cv2
import os

from sklearn.neighbors import KDTree



class VotingSegmentation:
    """ Voting based 3D point cloud segmentation given 2D masks & uv2pt lookups.
    """
    def __init__(
            self, npts, depth_hw, maskdir, uv2ptdir, nclasses, votes_file=None,
        ):
        """
        Args:
            npts (int): total points.
            depth_hw (tuple[int]): (height, width) - depth image resolution.
            maskdir (str): path to segmentation masks directory.
            uv2ptdir (str): path to uv2pt lookups directory.
            nclasses (int): number of classes.
            votes_file (str): direname/filename.npy - path to precomputed votes numpy file.

            Note:
                singe point from single mask can be voted to multiple classes
                due to sparsification(hence dense point merging).
        """
        if votes_file is None:
            self.npts = npts
            self.depth_hw = depth_hw
            self.nclasses = nclasses
            self.votes = np.zeros((npts, nclasses+1))

            self.mask_files, self.uv2pt_files = self._get_filenames(maskdir, uv2ptdir)
            self.nframes = len(self.mask_files)
        else:
            self.votes = np.load(votes_file)
            self.nclasses = self.votes.shape[1]

    def _get_filenames(self, maskdir, uv2ptdir):
        """ Private func to get mask and uv2pt lookup filenames.
        """
        maskdir = Path(maskdir)
        uv2ptdir = Path(uv2ptdir)
        mask_names = set([name.stem for name in maskdir.iterdir() if name.is_file()])
        uv2pt_names = set([name.stem for name in uv2ptdir.iterdir() if name.is_file()])
        mask_ext = next(maskdir.glob(f'{next(iter(mask_names))}.*')).suffix
        uv2pt_ext = next(uv2ptdir.glob(f'{next(iter(uv2pt_names))}.*')).suffix
        names = mask_names & uv2pt_names
        mask_files = [(maskdir/name).with_suffix(mask_ext) for name in names]
        uv2pt_files = [(uv2ptdir/name).with_suffix(uv2pt_ext) for name in names]
        return mask_files, uv2pt_files

    def _read_data(self, idx):
        """ Function to read segmentation mask and uv2pt lookup.

        Args:
            idx (int): frame index.

        Returns:
            np.ndarray[uint8]: [rgb_h, rgb_w] - segmentation mask.
            np.ndarray[int32]: [depth_h*depth_w, ] - uv2pt lookup.
        """
        mask = cv2.imread(str(self.mask_files[idx]), 0)
        uv2pt = np.load(self.uv2pt_files[idx])
        return mask, uv2pt

    def zero(self):
        """ Function to reinitialize state variables.
        """
        self.votes = np.zeros_like(self.votes)

    def vote(self, resize=True, verbose=False, filename=None):
        """ Funtion to vote corresponding points given semantic masks.

        Args:
            resize (bool): resize masks to depth_hw.
            verbose (bool): print progress.
            filename (str): write votes into .npy "filename".

        Returns:
            np.ndarray: [self.npts, nclasses] - votes.
        """
        h, w = self.depth_hw

        if verbose: print('voting ... ')
        for i in range(self.nframes):
            if verbose: print(f'frame/total = {i + 1}/{self.nframes}, progress = {((i + 1)*100/self.nframes):.3}%')

            mask, uv2pt = self._read_data(i)
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST) if resize else mask
            mask = mask.reshape(-1)
            valid = uv2pt != -1

            if valid.any():
                self.votes[uv2pt[valid], mask[valid]] += 1

        if filename is not None:
            Path(filename).parent.mkdir(exist_ok=True, parents=True)
            np.save(filename, self.votes)

        return self.votes

    def segment(self, threshold=0.5, filter_classes=None, votes=None):
        """ Function classify points given votes.

        Args:
            threshold (float): segmentation confidence threshold. class_vote/total_votes > threshold.
            filter_classes (tuple): list of classes to consider for segmentation.
            votes (np.ndarray): [self.npts, nclasses] - votes.

        Returns:
            np.ndarray: [self.npts, ] - point classes.
        """
        votes = self.votes if votes is None else votes
        votes = self.vote() if votes is None else votes

        total = votes.sum(-1)
        votes = votes if filter_classes is None else votes[:, filter_classes]
        valid = total > 0

        point_classes = np.argmax(votes, axis=1)
        point_maxes = votes[np.arange(len(votes)), point_classes]
        point_classes[np.logical_not(valid)] = self.nclasses

        prob = point_maxes[valid]/total[valid]
        less_confident = np.where(valid)[0][prob < threshold]
        point_classes[less_confident] = self.nclasses
        point_classes[point_maxes == 0] = self.nclasses

        if filter_classes is not None:
            for i, cls_ in enumerate(filter_classes):
                point_classes[point_classes == i] = cls_

        return point_classes


class PointVotingSegmentation:
    """
        DEPRECATED
    """
    def __init__(
            self, tofcameradata, sparse_points, depth_hw,
            maskdir, nclasses, prefix='', extension='png', zfill=2,
            votes_file=None,
        ):
        """ 3D Point Framewise Segmentation.

        Args:
            tofcameradata (list[dict]): [nframes, ...] - tooliqa tofcameradata.
            sparse_points (np.ndarray[float]): [M, 3] - sparse points.
            depth_hw (tuple[int]): (height, width).
            radius (float): neighbour search radius.
            threshold (float): voting threshold. class_vote/total_votes > threshold.
            filter_classes (tuple): list of classes to consider for voting.
            maskdir (str): path to mask directory.
            nclasses (int): number of classes.
            prefix (str): mask images prefix.
            extension (str): mask extension.
            zfill (int): zeros to filled in mask name.
            votes_file (str): direname/filename.npy - path to precomputed votes numpy file.

            Note:
                singe point from single mask can be voted to multiple classes
                due to sparsification(hence dense point merging).
        """
        if votes_file is None:
            self.nclasses = nclasses
            self.depth_hw = depth_hw
            self.tofcameradata = tofcameradata
            self.tree = KDTree(sparse_points, leaf_size=2)
            self.votes = np.zeros((len(sparse_points), nclasses + 1))

            self.maskdir = maskdir
            self.prefix = prefix
            self.ext = extension
            self.zfill = zfill
        else:
            self.votes = np.load(votes_file)
            self.nclasses = self.votes.shape[1] - 1

    def zero(self):
        """ Function to reinitialize state variables.
        """
        self.votes = np.zeros_like(self.votes)

    @classmethod
    def read_mask(cls, name, dirname='./', prefix='', extension='png', zfill=0):
        """ Function to read segmentation mask.

        Args:
            names (str/int): filename.
            dirname (str): path to directory.
            prefix (str): common prefixes.
            extension (str): file extension.
            zfill (int): fill zeros.

        Returns:
            np.ndarray: [H, W] - image - (dirname/prefixname.extension).

        """
        name = str(name)
        filename = os.path.join(dirname, prefix + name.zfill(zfill) + '.' + extension)
        return  cv2.imread(filename, 0) if os.path.isfile(filename) else None

    def get_nns(self, query_points, radius=0.01):
        """ Funtion to nearest sparse point given query points.

        Args:
            query_points (np.ndarray[float]): [Q, 3] - query points.
            radius (float): nn distance.

        Returns:
            np.ndarray: nns - [Q, frequency].reshape(-1) - sparse point cloud point indices.
            np.ndarray: frequency - [Q, ] - per pixel/query point counts.
        """
        nns = self.tree.query_radius(query_points, r=radius)
        frequency = np.array([len(idx) for idx in nns])
        nns = np.hstack(nns).astype(np.int32)
        return nns, frequency

    def vote(self, frame_numbers=None, skip=1, radius=0.01, resize=True, filename=None, verbose=False):
        """ Funtion to vote corresponding points given semantic masks.

        Args:
            frame_numbers (list[int]): frame numbers to be considered for voting.
            skip: number of frames to skip.
            radius (float): nearest neighbour distance.
            resize (bool): resize masks to depth_hw.
            filename (str): dirname/filename.npy path for storing votes.
            verbose (bool): print progress.

        Returns:
            np.ndarray: [N, nclasses + 1] - votes.
        """
        h, w = self.depth_hw
        frame_numbers = np.arange(len(self.tofcameradata)) if frame_numbers is None else frame_numbers
        total = len(frame_numbers)//skip

        if verbose: print('Framewise voting ... ')
        for i, idx in enumerate(frame_numbers[::skip]):
            data = self.tofcameradata[idx]
            if verbose: print(f'frame/total = {i + 1}/{total}, progress = {((i + 1)*100/total):.3}%')

            query_points = data['modPoints']
            fnum = int(data['frameNumber'])

            mask = self.read_mask(fnum, self.maskdir, self.prefix, self.ext, self.zfill)
            if mask is None: continue
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST) if resize else mask
            mask = mask.reshape(-1)

            point_indices, frequency = self.get_nns(query_points, radius)
            if point_indices.shape[0]:
                self.votes[point_indices, np.repeat(mask, frequency)] += 1
                self.votes[point_indices, -1] += 1

        if filename is not None:
            if verbose: print('writing file ... ')
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            np.save(filename, self.votes)

        return self.votes

    def segment(self, threshold, filter_classes=None, votes=None):
        """ Function classify points given votes.
        Args:
            threshold (float): voting threshold. class_vote/total_votes > threshold.
            filter_classes (tuple): list of classes to consider for voting.
            votes (np.ndarray): [N, nclasses + 1] - votes.

        Returns:
            np.ndarray: [N, ] - point classes.
        """
        votes = self.votes if votes is None else votes

        total = votes[:, -1]

        if filter_classes is not None:
            masked_votes = votes[:, filter_classes]
        else:
            masked_votes = votes[:, :-1]

        valid = total > 0
        point_classes = np.argmax(masked_votes, axis=1)
        point_maxes = masked_votes[np.arange(len(masked_votes)), point_classes]
        point_classes[np.logical_not(valid)] = self.nclasses

        less_confident = np.where(valid)[0][(point_maxes[valid]/total[valid]) < threshold]
        point_classes[less_confident] = self.nclasses
        point_classes[point_maxes == 0] = self.nclasses

        if filter_classes is not None:
            for i, cls_ in enumerate(filter_classes):
                point_classes[point_classes == i] = cls_

        return point_classes
