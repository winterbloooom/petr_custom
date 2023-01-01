# Copyright (c) Hikvision Research Institute. All rights reserved.
import mmcv
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Circle
from mmdet.core.visualization import color_val_matplotlib
from mmdet.core import bbox_mapping_back, multiclass_nms
from mmdet.models.detectors.single_stage import SingleStageDetector
from mmdet.models.detectors.detr import DETR

from opera.core.keypoint import bbox_kpt2result, kpt_mapping_back
from ..builder import DETECTORS


@DETECTORS.register_module()
class PETR(DETR):
    """Implementation of `End-to-End Multi-Person Pose Estimation with
    Transformers`"""

    def __init__(self, *args, **kwargs):
        super(DETR, self).__init__(*args, **kwargs)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_keypoints,
                      gt_areas,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box.
            gt_keypoints (list[Tensor]): Each item are the truth keypoints for
                each image in [p^{1}_x, p^{1}_y, p^{1}_v, ..., p^{K}_x,
                p^{K}_y, p^{K}_v] format.
            gt_areas (list[Tensor]): mask areas corresponding to each box.
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        print(f"img: {img.shape}") #torch.Size([2, 3, 965, 976]) 식 -> 한 배치에 이미지 2개인 듯
        # print(f"img_metas: {img_metas[0]['ori_filename']}, {img_metas[1]['ori_filename']}")
        """
        [{'filename': './dataset/public/coco/images/train2017/000000239845.jpg', 'ori_filename': '000000239845.jpg', 
        'ori_shape': (480, 640, 3), 'img_shape': (496, 429, 3), 'pad_shape': (496, 429, 3), 
        'scale_factor': array([0.8330097 , 0.83361346, 0.8330097 , 0.83361346], dtype=float32), 
        'flip': False, 'flip_direction': None, 
        'img_norm_cfg': {'mean': array([123.675, 116.28 , 103.53 ], dtype=float32), 'std': array([58.395, 57.12 , 57.375], dtype=float32), 
        'to_rgb': True}}, 
        
        {'filename': './dataset/public/coco/images/train2017/000000118895.jpg', 'ori_filename': '000000118895.jpg', 
        'ori_shape': (427, 640, 3), 'img_shape': (965, 976, 3), 'pad_shape': (965, 976, 3), 
        'scale_factor': array([2.243678 , 2.2441862, 2.243678 , 2.2441862], dtype=float32), 
        'flip': True, 'flip_direction': 'horizontal', 
        'img_norm_cfg': {'mean': array([123.675, 116.28 , 103.53 ], dtype=float32), 'std': array([58.395, 57.12 , 57.375], dtype=float32), 
        'to_rgb': True}}]
        """
        # print(f"gt_bboxes: {type(gt_bboxes)} / {gt_bboxes}")
        """
        list
        [tensor([[   0.,    0., 1230., 1400.]]), 
        
        tensor([[392.8949, 196.9397, 505.2177, 386.6451],
        [ 68.2744, 342.0081, 132.1774, 441.8358],
        [478.2634, 195.0032, 532.8507, 317.1267]])]
        """
        # print(f"gt_labels: {type(gt_labels)} / {gt_labels}")
        """
        list
        [tensor([0]), 
        tensor([0, 0, 0])]
        """
        # print(f"gt_keypoints: {type(gt_keypoints)} / {gt_keypoints}")
        """
        list
        3*N개 -> 3*17=51개
        x y v(0이면 없음 1이면 ~~ 2이면 ~~인 그거)
        [tensor([[263.1613, 525.1581,   2.0000, 374.4463, 368.5465,   2.0000, 192.8926,
         396.4264,   2.0000, 632.7856, 367.5768,   2.0000,   0.0000,   0.0000,
           0.0000, 911.6061, 964.5998,   2.0000,  19.7961, 839.0504,   2.0000,
           0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
           0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
           0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
           0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
           0.0000,   0.0000]]), 
           
           tensor([[410.8066, 222.6606,   2.0000, 410.8351, 217.9274,   2.0000,   0.0000,
           0.0000,   0.0000, 417.9461, 215.6036,   2.0000,   0.0000,   0.0000,
           0.0000, 432.0326, 233.4390,   2.0000, 448.6344, 226.4391,   2.0000,
         425.8616, 276.0025,   2.0000,   0.0000,   0.0000,   0.0000, 406.7798,
         301.9211,   2.0000,   0.0000,   0.0000,   0.0000, 469.5178, 294.0164,
           2.0000, 484.9369, 287.0094,   2.0000, 424.3932, 323.3278,   2.0000,
         464.5439, 334.2204,   2.0000, 464.3868, 360.2533,   2.0000, 473.6492,
         393.4432,   1.0000],
        [ 75.2415, 359.0868,   2.0000,  76.4457, 355.5440,   2.0000,   0.0000,
           0.0000,   0.0000,  83.5567, 353.2202,   2.0000,   0.0000,   0.0000,
           0.0000,  95.3204, 363.9415,   2.0000,  82.2954, 366.2296,   2.0000,
         107.0697, 377.0294,   2.0000,  86.9195, 384.0078,   2.0000, 115.2565,
         392.4624,   2.0000,  86.8481, 395.8409,   2.0000, 106.9769, 392.4124,
           2.0000,  98.6688, 397.0956,   2.0000, 108.0169, 416.0858,   2.0000,
         111.5867, 412.5573,   2.0000, 125.6803, 429.2093,   2.0000, 123.2933,
         432.7450,   2.0000],
        [496.0746, 205.4252,   2.0000, 496.0889, 203.0585,   2.0000, 492.5334,
         204.2204,   2.0000,   0.0000,   0.0000,   0.0000, 485.4152, 207.7275,
           2.0000, 503.1285, 212.5679,   2.0000, 482.9568, 223.0963,   2.0000,
         510.0825, 236.2770,   1.0000, 487.5523, 245.6078,   1.0000, 515.9393,
         245.7792,   2.0000, 496.9147, 262.2313,   2.0000, 513.4952, 258.7813,
           2.0000, 504.0114, 262.2741,   2.0000, 515.6895, 287.1951,   2.0000,
         514.5067, 287.1880,   2.0000, 504.8944, 311.9803,   2.0000, 528.5145,
         318.0398,   1.0000]])]
        """
        # print(f"gt_areas: {type(gt_areas)} / {gt_areas}")
        """
        list
        [tensor([2139973.2500]), 
        tensor([9943.8857, 2317.5857, 2607.1943])]
        """
        # print(f"gt_bboxes_ignore: {type(gt_bboxes_ignore)} / {gt_bboxes_ignore}")
        """
        NoneType
        None
        """

        super(SingleStageDetector, self).forward_train(img, img_metas)
        x = self.extract_feat(img) # from backbone / 이미지 사이즈가 크면 간혹 여기서 죽기도 함
        # 아래 결과는 모두 img: torch.Size([2, 3, 909, 1028])일 때

        # print("************Feature Extracted!************") # 실행됨
        # print(f"type(x): {type(x)}") # tuple
        # print(f"len(x): {len(x)}") #4 -> 각각이 level

        # print(f"type(x[0]): {type(x[0])}") # torch.Tensor
        # print(f"len(x[0]): {len(x[0])}") # 2
        # print(f"shape(x[0]): {x[0].shape}") # torch.Size([2, 256, 114, 129]) -> 해상도 1/8

        # print(f"type(x[1]): {type(x[1])}") # torch.Tensor
        # print(f"len(x[1]): {len(x[1])}") # 2
        # print(f"shape(x[1]): {x[1].shape}") # torch.Size([2, 256, 57, 65]) -> 해상도 1/16
        
        # print(f"type(x[2]): {type(x[2])}") # torch.Tensor
        # print(f"len(x[2]): {len(x[2])}") # 2
        # print(f"shape(x[2]): {x[2].shape}") # torch.Size([2, 256, 29, 33]) -> 해상도 1/32

        # print(f"type(x[3]): {type(x[3])}") # torch.Tensor
        # print(f"len(x[3]): {len(x[3])}") # 2
        # print(f"shape(x[3]): {x[3].shape}") # torch.Size([2, 256, 15, 17]) -> 해상도 1/60 (64)

        # print(f"len(x[0][0]): {len(x[0][0])}") # 256 -> demension
        # print(f"len(x[0][1]): {len(x[0][1])}") # 256 -> demension
        # print(f"len(x[0][0]): {len(x[1][0])}") # 256 -> demension
        # print(f"len(x[0][0]): {len(x[2][0])}") # 256 -> demension
        # print(f"len(x[0][0]): {len(x[3][0])}") # 256 -> demension
        
        # import sys
        # sys.exit(0)

        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_keypoints,
                                              gt_areas, gt_bboxes_ignore)
            # bbox_head는 petr_head.py => losses의 type: dict
        """
        {'enc_loss_cls': tensor([2.2451], device='cuda:0', grad_fn=<MulBackward0>), 
        'enc_loss_kpt': tensor(7.9735, device='cuda:0', grad_fn=<MulBackward0>), 
        'loss_cls': tensor([1.8840], device='cuda:0', grad_fn=<MulBackward0>), 
        'loss_kpt': tensor(8.4770, device='cuda:0', grad_fn=<MulBackward0>), 
        'loss_oks': tensor(4.8181, device='cuda:0', grad_fn=<MulBackward0>), 
        'd0.loss_cls': tensor([2.3254], device='cuda:0', grad_fn=<MulBackward0>), 
        'd0.loss_kpt': tensor(8.4770, device='cuda:0', grad_fn=<MulBackward0>), 
        'd0.loss_oks': tensor(4.8181, device='cuda:0', grad_fn=<MulBackward0>), 
        'd1.loss_cls': tensor([2.1434], device='cuda:0', grad_fn=<MulBackward0>), 
        'd1.loss_kpt': tensor(8.4770, device='cuda:0', grad_fn=<MulBackward0>), 
        'd1.loss_oks': tensor(4.8181, device='cuda:0', grad_fn=<MulBackward0>), 
        'loss_hm': tensor(152.6593, device='cuda:0', grad_fn=<MulBackward0>), 
        'd0.loss_kpt_refine': tensor(9.6879, device='cuda:0', grad_fn=<MulBackward0>), 
        'd0.loss_oks_refine': tensor(7.2271, device='cuda:0', grad_fn=<MulBackward0>), 
        'd1.loss_kpt_refine': tensor(9.6879, device='cuda:0', grad_fn=<MulBackward0>), 
        'd1.loss_oks_refine': tensor(7.2271, device='cuda:0', grad_fn=<MulBackward0>)}
        """
        # print(f"losses: {losses.shape} / {losses}") # TODO 잠시 멈춤. dict 요소에는 shape 속성 없어 에러 나서.
        
        return losses

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`.
        """
        warnings.warn('Warning! MultiheadAttention in DETR does not '
                      'support flops computation! Do not use the '
                      'results in your papers!')

        batch_size, _, height, width = img.shape
        dummy_img_metas = [
            dict(
                batch_input_shape=(height, width),
                img_shape=(height, width, 3),
                scale_factor=(1., 1., 1., 1.)) for _ in range(batch_size)
        ]
        x = self.extract_feat(img)
        outs = self.bbox_head(x, img_metas=dummy_img_metas)
        bbox_list = self.bbox_head.get_bboxes(
            *outs, dummy_img_metas, rescale=True)
        return bbox_list

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test time augmentation.

        Args:
            img (list[torch.Tensor]): List of multiple images.
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox and keypoint results of each image
                and classes. The outer list corresponds to each image.
                The inner list corresponds to each class.
        """
        batch_size = len(img_metas)
        assert batch_size == 1, 'Currently only batch_size 1 for inference ' \
            f'mode is supported. Found batch_size {batch_size}.'
        
        feat = self.extract_feat(img)
        results_list = self.bbox_head.simple_test(
            feat, img_metas, rescale=rescale)

        bbox_kpt_results = [
            bbox_kpt2result(det_bboxes, det_labels, det_kpts,
                            self.bbox_head.num_classes)
            for det_bboxes, det_labels, det_kpts in results_list
        ]
        return bbox_kpt_results

    def merge_aug_results(self, aug_bboxes, aug_kpts, aug_scores, img_metas):
        """Merge augmented detection bboxes and keypoints.

        Args:
            aug_bboxes (list[Tensor]): shape (n, 4).
            aug_kpts (list[Tensor] or None): shape (n, K, 2).
            img_metas (list): meta information.

        Returns:
            tuple: (bboxes, kpts, scores).
        """
        recovered_bboxes = []
        recovered_kpts = []
        for bboxes, kpts, img_info in zip(aug_bboxes, aug_kpts, img_metas):
            img_shape = img_info[0]['img_shape']
            scale_factor = img_info[0]['scale_factor']
            flip = img_info[0]['flip']
            flip_direction = img_info[0]['flip_direction']
            bboxes = bbox_mapping_back(bboxes, img_shape, scale_factor, flip,
                                       flip_direction)
            kpts = kpt_mapping_back(kpts, img_shape, scale_factor, flip,
                                    flip_direction)
            recovered_bboxes.append(bboxes)
            recovered_kpts.append(kpts)
        bboxes = torch.cat(recovered_bboxes, dim=0)
        kpts = torch.cat(recovered_kpts, dim=0)
        if aug_scores is None:
            return bboxes, kpts
        else:
            scores = torch.cat(aug_scores, dim=0)
            return bboxes, kpts, scores

    def aug_test(self, imgs, img_metas, rescale=False):
        feats = self.extract_feats(imgs)
        aug_bboxes = []
        aug_scores = []
        aug_kpts = []
        for x, img_meta in zip(feats, img_metas):
            # only one image in the batch
            outs = self.bbox_head(x, img_meta)
            bbox_list = self.bbox_head.get_bboxes(
                *outs, img_meta, rescale=False)

            for det_bboxes, det_labels, det_kpts in bbox_list:
                aug_bboxes.append(det_bboxes[:, :4])
                aug_scores.append(det_bboxes[:, 4])
                aug_kpts.append(det_kpts[..., :2])

        merged_bboxes, merged_kpts, merged_scores = self.merge_aug_results(
            aug_bboxes, aug_kpts, aug_scores, img_metas)

        merged_scores = merged_scores.unsqueeze(1)
        padding = merged_scores.new_zeros(merged_scores.shape[0], 1)
        merged_scores = torch.cat([merged_scores, padding], dim=-1)
        det_bboxes, det_labels, keep_inds = multiclass_nms(
            merged_bboxes,
            merged_scores,
            self.test_cfg.score_thr,
            self.test_cfg.nms,
            self.test_cfg.max_per_img,
            return_inds=True)
        det_kpts = merged_kpts[keep_inds]
        det_kpts = torch.cat(
            (det_kpts, det_kpts.new_ones(det_kpts[..., :1].shape)), dim=2)

        bbox_kpt_results = [
            bbox_kpt2result(det_bboxes, det_labels, det_kpts,
                            self.bbox_head.num_classes)
        ]
        return bbox_kpt_results

    def show_result(self,
                    img,
                    result,
                    score_thr=0.3,
                    bbox_color=(72, 101, 241),
                    text_color=(72, 101, 241),
                    mask_color=None,
                    thickness=2,
                    font_size=10,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (Tensor or tuple): The results to draw over `img`
                bbox_result or (bbox_result, segm_result).
            score_thr (float, optional): Minimum score of bboxes to be shown.
                Default: 0.3.
            bbox_color (str or tuple(int) or :obj:`Color`):Color of bbox lines.
               The tuple of color should be in BGR order. Default: 'green'.
            text_color (str or tuple(int) or :obj:`Color`):Color of texts.
               The tuple of color should be in BGR order. Default: 'green'.
            mask_color (None or str or tuple(int) or :obj:`Color`):
               Color of masks. The tuple of color should be in BGR order.
               Default: None.
            thickness (int): Thickness of lines. Default: 2.
            font_size (int): Font size of texts. Default: 13.
            win_name (str): The window name. Default: ''.
            wait_time (float): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            img (Tensor): Only if not `show` or `out_file`.
        """
        img = mmcv.imread(img)
        img = img.copy()
        if isinstance(result, tuple):
            bbox_result, keypoint_result = result
            segm_result = None
        else:
            bbox_result, segm_result, keypoint_result = result, None, None
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        # draw segmentation masks
        segms = None
        if segm_result is not None and len(labels) > 0:  # non empty
            segms = mmcv.concat_list(segm_result)
            if isinstance(segms[0], torch.Tensor):
                segms = torch.stack(segms, dim=0).detach().cpu().numpy()
            else:
                segms = np.stack(segms, axis=0)
        # draw keypoints
        keypoints = None
        if keypoint_result is not None:
            keypoints = np.vstack(keypoint_result)
        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False
        # draw bounding boxes
        img = self.imshow_det_bboxes(
            img,
            bboxes,
            labels,
            segms,
            keypoints,
            class_names=self.CLASSES,
            score_thr=score_thr,
            bbox_color=bbox_color,
            text_color=text_color,
            mask_color=mask_color,
            thickness=thickness,
            font_size=font_size,
            win_name=win_name,
            show=show,
            wait_time=wait_time,
            out_file=out_file)

        if not (show or out_file):
            return img

    def imshow_det_bboxes(self,
                          img,
                          bboxes,
                          labels,
                          segms=None,
                          keypoints=None,
                          class_names=None,
                          score_thr=0,
                          bbox_color='green',
                          text_color='green',
                          mask_color=None,
                          thickness=2,
                          font_size=8,
                          win_name='',
                          show=True,
                          wait_time=0,
                          out_file=None):
        """Draw bboxes and class labels (with scores) on an image.

        Args:
            img (str or ndarray): The image to be displayed.
            bboxes (ndarray): Bounding boxes (with scores), shaped (n, 4) or
                (n, 5).
            labels (ndarray): Labels of bboxes.
            segms (ndarray or None): Masks, shaped (n,h,w) or None.
            keypoints (ndarray): keypoints (with scores), shaped (n, K, 3).
            class_names (list[str]): Names of each classes.
            score_thr (float): Minimum score of bboxes to be shown. Default: 0.
            bbox_color (str or tuple(int) or :obj:`Color`):Color of bbox lines.
                The tuple of color should be in BGR order. Default: 'green'.
                text_color (str or tuple(int) or :obj:`Color`):Color of texts.
                The tuple of color should be in BGR order. Default: 'green'.
            mask_color (str or tuple(int) or :obj:`Color`, optional):
                Color of masks. The tuple of color should be in BGR order.
                Default: None.
            thickness (int): Thickness of lines. Default: 2.
            font_size (int): Font size of texts. Default: 13.
            show (bool): Whether to show the image. Default: True.
            win_name (str): The window name. Default: ''.
            wait_time (float): Value of waitKey param. Default: 0.
            out_file (str, optional): The filename to write the image.
                Default: None.

        Returns:
            ndarray: The image with bboxes drawn on it.
        """
        assert bboxes.ndim == 2, \
            f' bboxes ndim should be 2, but its ndim is {bboxes.ndim}.'
        assert labels.ndim == 1, \
            f' labels ndim should be 1, but its ndim is {labels.ndim}.'
        assert bboxes.shape[0] == labels.shape[0], \
            'bboxes.shape[0] and labels.shape[0] should have the same length.'
        assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5, \
            f' bboxes.shape[1] should be 4 or 5, but its {bboxes.shape[1]}.'
        img = mmcv.imread(img).astype(np.uint8)

        if score_thr > 0:
            assert bboxes.shape[1] == 5
            scores = bboxes[:, -1]
            inds = scores > score_thr

            print(f"inds: {inds}") #TODO
            
            bboxes = bboxes[inds, :]
            labels = labels[inds]
            if segms is not None:
                segms = segms[inds, ...]
            if keypoints is not None:
                keypoints = keypoints[inds, ...]

            print(f"bboxes over thre: \n{bboxes}")
            print(f"keypoints over thre: \n{keypoints}")

        num_keypoint = keypoints.shape[1]
        if num_keypoint == 14:
            colors_hp = [(169, 209, 142), (255, 255, 0), (169, 209, 142),
                         (255, 255, 0), (169, 209, 142), (255, 255, 0),
                         (0, 176, 240), (252, 176, 243), (0, 176, 240),
                         (252, 176, 243), (0, 176, 240), (252, 176, 243),
                         (236, 6, 124), (236, 6, 124)]
        elif num_keypoint == 17:
            colors_hp = [(236, 6, 124), (236, 6, 124), (236, 6, 124),
                         (236, 6, 124), (236, 6, 124), (169, 209, 142),
                         (255, 255, 0), (169, 209, 142), (255, 255, 0),
                         (169, 209, 142), (255, 255, 0), (0, 176, 240),
                         (252, 176, 243), (0, 176, 240), (252, 176, 243),
                         (0, 176, 240), (252, 176, 243)]
        else:
            raise ValueError(f'unsupported keypoint amount {num_keypoint}')
        colors_hp = [color[::-1] for color in colors_hp]
        colors_hp = [color_val_matplotlib(color) for color in colors_hp]

        if num_keypoint == 14:
            edges = [
                [0, 2],
                [2, 4],
                [1, 3],
                [3, 5],  # arms
                [0, 1],
                [0, 6],
                [1, 7],  # body
                [6, 8],
                [8, 10],
                [7, 9],
                [9, 11],  # legs
                [12, 13]
            ]  # neck
            ec = [(169, 209, 142),
                  (169, 209, 142), (255, 255, 0), (255, 255, 0), (255, 102, 0),
                  (0, 176, 240), (252, 176, 243), (0, 176, 240), (0, 176, 240),
                  (252, 176, 243), (252, 176, 243), (236, 6, 124)]
        elif num_keypoint == 17:
            edges = [
                [0, 1],
                [0, 2],
                [1, 3],
                [2, 4],  # head
                [5, 7],
                [7, 9],
                [6, 8],
                [8, 10],  # arms
                [5, 6],
                [5, 11],
                [6, 12],  # body
                [11, 13],
                [13, 15],
                [12, 14],
                [14, 16]
            ]  # legs
            ec = [(236, 6, 124), (236, 6, 124), (236, 6, 124), (236, 6, 124),
                  (169, 209, 142),
                  (169, 209, 142), (255, 255, 0), (255, 255, 0), (255, 102, 0),
                  (0, 176, 240), (252, 176, 243), (0, 176, 240), (0, 176, 240),
                  (252, 176, 243), (252, 176, 243)]
        else:
            raise ValueError(f'unsupported keypoint amount {num_keypoint}')
        ec = [color[::-1] for color in ec]
        ec = [color_val_matplotlib(color) for color in ec]

        img = mmcv.bgr2rgb(img)
        width, height = img.shape[1], img.shape[0]
        img = np.ascontiguousarray(img)

        EPS = 1e-2
        fig = plt.figure(win_name, frameon=False)
        plt.title(win_name)
        canvas = fig.canvas
        dpi = fig.get_dpi()
        # add a small EPS to avoid precision lost due to matplotlib's truncation
        # (https://github.com/matplotlib/matplotlib/issues/15363)
        fig.set_size_inches((width + EPS) / dpi, (height + EPS) / dpi)

        # remove white edges by set subplot margin
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax = plt.gca()
        ax.axis('off')

        polygons = []
        color = []
        for i, (bbox, label, kpt) in enumerate(zip(bboxes, labels, keypoints)):
            bbox_int = bbox.astype(np.int32)
            poly = [[bbox_int[0], bbox_int[1]], [bbox_int[0], bbox_int[3]],
                    [bbox_int[2], bbox_int[3]], [bbox_int[2], bbox_int[1]]]
            np_poly = np.array(poly).reshape((4, 2))
            # polygons.append(Polygon(np_poly))
            # color.append(bbox_color)
            # label_text = class_names[
            #     label] if class_names is not None else f'class {label}'
            # if len(bbox) > 4:
            #     label_text += f'|{bbox[-1]:.02f}'
            # get left-top corner of all keypoints
            bbox_int[0] = np.floor(kpt[:, 0].min()).astype(np.int32)
            bbox_int[1] = np.floor(kpt[:, 1].min() - 30).astype(np.int32)
            label_text = f'{bbox[-1]:.02f}'
            # ax.text(
            #     bbox_int[0],
            #     bbox_int[1],
            #     f'{label_text}',
            #     bbox={
            #         'facecolor': 'black',
            #         'alpha': 0.8,
            #         'pad': 0.7,
            #         'edgecolor': 'none'
            #     },
            #     color=text_color,
            #     fontsize=font_size,
            #     verticalalignment='top',
            #     horizontalalignment='left')
            for j in range(kpt.shape[0]): # 17개
                ax.add_patch(
                    Circle(
                        xy=(kpt[j, 0], kpt[j, 1]),
                        radius=2,
                        color=colors_hp[j]))
            for j, e in enumerate(edges):
                poly = [[kpt[e[0], 0], kpt[e[0], 1]],
                        [kpt[e[1], 0], kpt[e[1], 1]]]
                np_poly = np.array(poly).reshape((2, 2))
                polygons.append(Polygon(np_poly))
                color.append(ec[j])
            if segms is not None:
                color_mask = mask_colors[labels[i]]
                mask = segms[i].astype(bool)
                img[mask] = img[mask] * 0.5 + color_mask * 0.5

        plt.imshow(img)

        p = PatchCollection(
            polygons, facecolor='none', edgecolors=color, linewidths=thickness)
        ax.add_collection(p)

        stream, _ = canvas.print_to_buffer()
        buffer = np.frombuffer(stream, dtype='uint8')
        img_rgba = buffer.reshape(height, width, 4)
        rgb, alpha = np.split(img_rgba, [3], axis=2)
        img = rgb.astype('uint8')
        img = mmcv.rgb2bgr(img)

        if show:
            # We do not use cvc2 for display because in some cases, opencv will
            # conflict with Qt, it will output a warning: Current thread
            # is not the object's thread. You an refer to
            # https://github.com/opencv/opencv-python/issues/46 for details
            if wait_time == 0:
                plt.show()
            else:
                plt.show(block=False)
                plt.pause(wait_time)
        if out_file is not None:
            mmcv.imwrite(img, out_file)

        plt.close()
        return img
