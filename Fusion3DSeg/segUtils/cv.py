import numpy as np





class CVSegmentation:
    def __init__(self, classes, adj):
        """ classic segmentation algorithms

        Args:
            classes (np.ndarray[int]): [N, ] - point classes.
            adj (list[set]): [N, ...] - adjcency list.
        """
        self.classes = classes
        self.adj = adj

    @staticmethod
    def _floodfill_level(point, adj, max_level=50, inq=None):
        """ Fucntion to find all connected points around a given point within given distance(level).

        Args:
            point (int): seed index of point in point cloud. point < N.
            adj (list[set]): [N, ...] - adjecency list.
            max_level (int): max recursion/depth level.
            inq (np.ndarray[bool]): [True if inq else False for node in point_cloud].

        Returns:
            list: [M, ] - list of cluster points.
        """
        inq = np.zeros(len(adj), bool) if inq is None else inq
        points_q = [point]
        inq[point] = True
        levels = [1]
        cluster = []
        mask = mask if mask is not None else np.ones(len(adj), bool)

        while points_q:
            point = points_q.pop(0)
            level = levels.pop(0)
            if (level == max_level): continue
            cluster.append(point)
            neighbours = adj[point]
            non_inq = [n for n in neighbours if (not inq[n]) and mask[n]]
            points_q += non_inq
            levels += [level + 1 for _ in non_inq]
            inq[non_inq] = True

        return cluster

    @staticmethod
    def _floodfill_class(point, adj, classes, inq=None):
        """ Fucntion to find all connected points around a given point with same class.

        Args:
            point (int): seed index of point in point cloud. point < N.
            adj (list[set]): [N, ...] - adjecency list.
            classes (np.ndarray[int]): [N, ] - point classes.
            inq (np.ndarray[bool]): [True if inq else False for node in point_cloud].

        Returns:
            int: seed class.
            int: no of cluster points.
            np.ndarray(int): [M, ] - cluster points.
            np.ndarray(bool): [N, ] - boolean array of boundary points.
        """
        inq = np.zeros(len(classes), bool) if inq is None else inq
        points_q = [point]
        seed_class = classes[point]
        inq[point] = True
        cluster = []
        boundary = np.zeros(len(classes), bool)
        parents = np.full_like(classes, -1)
        parents[point] = point

        while points_q:
            point = points_q.pop(0)
            if classes[point] != seed_class:
                boundary[parents[point]] = True
                continue
            cluster.append(point)
            neighbours = adj[point]
            non_inq = [n for n in neighbours if (not inq[n])]
            points_q += non_inq
            parents[non_inq] = point
            inq[non_inq] = True

        cluster = np.array(cluster)
        return seed_class, len(cluster), cluster, boundary

    @staticmethod
    def _floodfill_color(point, adj, colors, ids, threshold, max_level=10, mask=None):
        """ Fucntion to find all connected points around a given point with similar color.

        Algo:
            for neighbour in neighbours:
                if criterion(color[neighbour]) and criterion([class[neighbour]]) and criterion(level[neighbour]):
                    cluster.append(neighbour)
            criterion[class[neighbour]] = mask[neighbour]

        Args:
            point (int): seed index of point in point cloud. point < N.
            adj (list[set]): [N, ...] - adjecency list.
            colors (np.ndarray[float]): [N, 3] - point colors.
            ids (np.ndarray[ids]): [N, ] - points instance ids.
            threshold (np.ndarray[float]): [r, g, b] - color threshold.
            max_level (int): max recursion/depth level.
            mask (np.ndarray[bool]): [N, ] - neutral mask.

        Returns:
            np.ndarray[int]: [N, ] - point instance ids.
        """
        n = len(ids)
        inq = np.zeros(n, bool)
        mask = mask if mask is not None else np.ones(n, bool)
        points_q = [point]
        inq[point] = True
        levels = [1]

        seed_id = ids[point]
        sma = colors[point]
        npts = 0

        while points_q:
            point = points_q.pop(0)
            level = levels.pop(0)
            clr = colors[point]
            if (level == max_level): continue
            if (np.abs(sma - clr) > threshold).any(): continue

            npts += 1
            sma = sma + (clr - sma)/npts

            ids[point] = seed_id
            neighbours = adj[point]

            non_inq = [n for n in neighbours if (not inq[n]) and mask[n]]
            points_q += non_inq
            levels += [level + 1 for _ in non_inq]
            inq[non_inq] = True

        return ids

    def _get_clusters(self, cluster_points, max_level=50, inq=None):
        """ Function to cluster connected points and find major cluster

        Args:
            points (np.ndarray): point indices.
            max_level (int): max recursion/distance.
            inq (np.ndarray): [True if inq else False for node in point_cloud].

        Returns:
            list[np.ndarray]: list of connected point clusters.
        """
        clusters = []
        mask = np.zeros(len(self.adj), dtype=bool)
        mask[cluster_points] = True
        all_points = np.arange(len(self.adj))
        remaining_points = all_points[mask]

        while len(remaining_points):
            point = remaining_points[0]
            cluster = self._floodfill_level(point, self.adj, max_level, inq=None)
            clusters.append(cluster)
            mask[cluster] = False
            remaining_points = all_points[mask]

        return clusters

    @staticmethod
    def merge_classes(classes, source, destination):
        """ Function to merge classes

        Args:
            classes (np.ndarray): [N, ] - classes.
            source (tuple[int]): [M, ] - source classes.
            destination (tuple[int]): [M, ] - destination classes.

        Returns:
            np.ndarray: [N, ] - classes.
        """
        for fcls, tcls in zip(source, destination):
            classes[classes == fcls] = tcls
        return classes

    @staticmethod
    def get_semantic_object_ids(idinfo):
        """ Funciton to get ids associated with object classes and semantic classes.
        Args:
            idinfo (list[dict]): [M, ] - id info - {'id':id, 'isthing':object_or_not, 'category_id':class, 'area':npts}.

        Returns:
            list[int]: [P, ] - semantic ids list.
            list[int]: [Q, ] - object ids list. P + Q = M.
        """
        objectids, semanticids = [], []
        for info in idinfo:
            if info['isthing']:
                objectids.append(info['id'])
            else:
                semanticids.append(info['id'])
        return semanticids, objectids

    @staticmethod
    def get_objects(ids, objectids):
        """ Funtion to get points with object ids.

        Args:
            ids (np.ndarray[int]): [N, ] - points instance ids.
            objectids (list[int]): [M, ] - list of object ids.

        Returns:
            np.ndarray[bool]: points with objects mask.
        """
        objects = np.zeros(len(ids), bool)
        for id_ in objectids:
            objects[ids == id_] = True
        return objects

    @staticmethod
    def get_classes(ids, idinfo):
        """ Funciton to get point classes given instance ids and idinfo.

        Args:
            ids (np.ndarray[int]): [N, ] - point instance ids.
            idinfo (list[dict]): [M, ] - instancepoint_classes id info.

        Returns:
            np.ndarray[int]: [N, ] - point classes.
        """
        classes = np.zeros_like(ids)
        for info in idinfo:
            classes[ids == info['id']] = info['category_id']
        return classes

    @staticmethod
    def get_ids_by_classes(idinfo, classes):
        """ Funciton to get ids with given class categorires.

        Args:
            idinfo (list[dict]): [M, ] - instancepoint_classes id info.
            classes (tuple[int]): [M, ] - list of categorires.

        Returns:
            list[list[int]] - [M, ...] - [[list of ids] for category in categories].
        """
        classids = [[] for _ in classes]
        for info in idinfo:
            id_, cls_ = info['id'], info['category_id']
            for i, category in enumerate(classes):
                if category == cls_:
                    classids[i].append(id_)
        return classids

    @staticmethod
    def merge_instances_by_classes(ids, idinfo, classes, clusters=None, boundaries=None):
        """ Funciton to merge instances of given classes.

        Args:
            ids (np.ndarray[int]): [N, ] - point instance ids.
            idinfo (list[dict]): [M, ] - {'id':id, 'isthing':object_or_not, 'category_id':class, 'area':npts}.
            classes (tuple[int]): [K, ] - category ids/classes.
            clusters (list[list[int]]): [M, ...] - [list of points for id in range(M)] - cluster points.
            boundaries (list[list[int]]): [M, ...] - [list of boundary points if instance elif semantic None for id in range(M)] - boundaries.

        Returns:
            int: M - total instance ids.
            np.ndarray[int]: [N, ] - point instance ids.
            list[dict]: [M, ] - id info - {'id':id, 'isthing':object_or_not, 'category_id':class, 'area':npts}.
            list[list[int]]: [M, ...] - [list of points for id in range(M)] - cluster points.
            list[list[int]]: [M, ...] - [list of boundary points if instance elif semantic None for id in range(M)] - boundaries.
        """

        outids = ids.copy()
        outidinfo, outclusters, outboundaries = [], [], []
        classids = [None for _ in classes]
        ninstances = 0
        for i, info in enumerate(idinfo):
            id_, cat = info['id'], info['category_id']
            outlier = True
            for j, cls_ in enumerate(classes):
                if cat == cls_:
                    if classids[j] is None:
                        classids[j] = ninstances
                        outids[ids == id_] = ninstances
                        ninstances += 1
                        outidinfo.append(info)
                        outclusters.append([clusters[i]])
                        outboundaries.append([boundaries[i]])
                    else:
                        clsid = classids[j]
                        outids[ids == id_] = clsid
                        outidinfo[clsid]['area'] += info['area']
                        outclusters[clsid].append(clusters[i])
                        outboundaries[clsid].append(boundaries[i])
                    outlier = False
                    break
            if outlier:
                outids[ids == id_] = ninstances
                ninstances += 1
                outidinfo.append(info)
                outclusters.append([clusters[i]])
                outboundaries.append([boundaries[i]])

        outclusters = [np.hstack(clstrs) for clstrs in outclusters]
        outboundaries = [np.hstack(clstrs) for clstrs in outboundaries]
        return ninstances + 1, outids, outidinfo, outclusters, outboundaries

    def instance_seperate(self, instance_classes=None, minimum_points=1):
        """ Function seperate non connected same class object ninstances.

        Args:
            instance_classes (tuple[int]): list of classes to consider for instance seperation.
            minimum_points (int): minimum number of points per cluster to consider it as a object
                                  otherwise rewrite its class 0.

        Returns:
            np.ndarray[int]: [M, ] - M instance ids.
            np.ndarray[int]: [N, ] - point instance ids.
            list[dict]: [M, ] - id info - {'id':id, 'isthing':object_or_not, 'category_id':class, 'area':npts}.
            list[list[int]]: [M, ...] - [list of points for id in range(M)] - cluster points.
            list[list[int]]: [M, ...] - [list of boundary points if instance elif semantic None for id in range(M)] - boundaries.
        """
        n = len(self.classes)
        all_points = np.arange(n)
        allclasses = np.unique(self.classes)
        ids = np.zeros_like(self.classes)
        semantics = np.zeros(n, bool)
        info, clusters, boundaries = [], [], []
        if instance_classes is None:
            instance_classes = allclasses
            ninstances = 0
        else:
            instance_classes = np.array(instance_classes)
            semantic_classes = np.setdiff1d(allclasses, instance_classes)
            ninstances = len(semantic_classes)
            for instid in range(ninstances):
                cls_ = semantic_classes[instid]
                mask = self.classes == cls_
                ids[mask] = instid
                idinfo = {'id':instid, 'isthing':False, 'category_id':int(cls_), 'area':int(mask.sum())}
                info.append(idinfo)
                semantics[mask] = True
                cluster = all_points[mask].copy()
                clusters.append(cluster)
                boundaries.append(None)
        for cls_ in instance_classes:
            mask = self.classes == cls_
            remaining_points = all_points[mask]
            while len(remaining_points):
                point = remaining_points[0]
                seed_class, area, cluster, boundary = self._floodfill_class(point, self.adj, self.classes, inq=None)
                category = 0 if area < minimum_points else cls_
                idinfo = {'id':ninstances, 'isthing':True, 'category_id':int(category), 'area':int(area)}
                ids[cluster] = ninstances
                info.append(idinfo)
                clusters.append(cluster)
                boundaries.append(boundary)
                ninstances += 1

                mask[cluster] = False
                remaining_points = all_points[mask]
                self.classes[cluster] = category
        ninstances, ids, info, clusters, boundaries =  self.merge_instances_by_classes(ids, info, (0, ), clusters, boundaries)
        return np.arange(ninstances), ids, info, clusters, boundaries

    def color_segment(self, colors, ids, seeds, threshold, neutral_ids=(0, ), max_level=10):
        """ Function post process segmentations with color information.

        Args:
            colors (np.ndarray[float]): [N, 3] - point colors.
            ids (np.ndarray[int]): [N, ] - point instance ids.
            seeds (np.ndarray[int]): [P, ] - seeds points to start color segmentation.
            threshold (tuple[float]): [r, g, b] - color threshold.
            neutral_ids (tuple[int]): [K, ] - neutral ids which can be overwritten.
            max_level (int): max recursion/depth till color segment.

        Returns:
            np.ndarray: [N, ] - point ids.

        Note:
            beta.
            not verified/tested.

        Todo:
            confidence based intersion handling color segmentation instead of first come first serve statergy.
        """
        n = len(ids)
        threshold = np.array(threshold) if not isinstance(threshold, (int, float)) else np.array([threshold, threshold, threshold])
        neutral_mask = np.zeros(n, bool)
        for id_ in neutral_ids:
            neutral_mask[ids == id_] = True

        for seed in seeds:
            seed_id = ids[seed]
            ids = self._floodfill_color(seed, self.adj, colors, ids, threshold, max_level, neutral_mask)
            neutral_mask[ids == seed_id] = False

        return ids


def split_into_instances(classes, adj, nclasses=133, instance_classes=None, minimum_points=1, verbose=False):
    """ Function seperate non connected same class object instances.

    Note:
        This function does not return cluster points, boudaries.
        Not compatible with color segmentation.
        Use CVSegmentation.intance_seperate to overcome this limitations.

    Args:
        classes (np.ndarray[int]): [N, ] - point classes.
        adj (list[set]): [N, ...] - adjcency list.
        nclasses (int): total categories.
        instance_classes (tuple[int]): list of classes to consider for instance seperation.
        minimum_points (int): minimum number of points per cluster to consider it as a object
                                otherwise rewrite its class 0.
        verbose (bool): print progress.

    Returns:
        np.ndarray[int]: [M, ] - M instance ids.
        np.ndarray[int]: [N, ] - point instance ids.
        list[dict]: [M, ] - id info - {'id':id, 'isthing':object_or_not, 'category_id':class, 'area':npts}.
        np.ndarray[int]: [N, ] - point classes updated.
    """
    def floodfill(point, adj, classes):
        inq = np.zeros(len(classes), bool)
        points_q = [point]
        seed_class = classes[point]
        inq[point] = True
        cluster = []
        while points_q:
            point = points_q.pop(0)
            if classes[point] != seed_class: continue
            cluster.append(point)
            neighbours = adj[point]
            non_inq = [n for n in neighbours if (not inq[n])]
            points_q += non_inq
            inq[non_inq] = True
        cluster = np.array(cluster)
        return seed_class, len(cluster), cluster

    n = len(classes)
    classes = classes.copy()
    all_points = np.arange(n)
    allclasses = np.unique(classes)
    ids = np.zeros_like(classes)
    info = []
    small_disjoint_id = None
    if instance_classes is None:
        instance_classes = allclasses
        ninstances = 0
        semantic_classes = []
        unclassified = instance_classes == nclasses
        if unclassified.any():
            instance_classes = instance_classes[np.logical_not(unclassified)]
            semantic_classes = [nclasses]
            ninstances = 1
    else:
        instance_classes = np.array(instance_classes)
        semantic_classes = np.setdiff1d(allclasses, instance_classes)
        ninstances = len(semantic_classes)

    if len(semantic_classes):
        for instid in range(ninstances):
            cls_ = semantic_classes[instid]
            mask = classes == cls_
            ids[mask] = instid
            idinfo = {'id':instid, 'isthing':False, 'category_id':int(cls_), 'area':int(mask.sum())}
            if cls_ == nclasses:
                small_disjoint_id = instid
            info.append(idinfo)

    for cls_ in instance_classes:
        if verbose: print('splitting class:', cls_)
        mask = classes == cls_
        remaining_points = all_points[mask]
        while len(remaining_points):
            point = remaining_points[0]
            seed_class, area, cluster = floodfill(point, adj, classes)

            if area < minimum_points:
                category = nclasses
                if small_disjoint_id is None:
                    small_disjoint_id = ninstances
                    idinfo = {'id':ninstances, 'isthing':True, 'category_id':int(category), 'area':0}
                    info.append(idinfo)
                    ninstances += 1
                info[small_disjoint_id]['area'] += area
                ids[cluster] = small_disjoint_id
            else:
                category = cls_
                idinfo = {'id':ninstances, 'isthing':True, 'category_id':int(category), 'area':int(area)}
                ids[cluster] = ninstances
                info.append(idinfo)
                ninstances += 1

            mask[cluster] = False
            remaining_points = all_points[mask]
            classes[cluster] = category
    return np.arange(ninstances), ids, info, classes