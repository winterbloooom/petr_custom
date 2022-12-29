# Copyright (c) Hikvision Research Institute. All rights reserved.
import math

import torch
import torch.nn as nn
from torch.nn.init import normal_
from mmcv.cnn import constant_init, xavier_init
from mmcv.cnn.bricks.transformer import (BaseTransformerLayer,
                                         TransformerLayerSequence)
from mmcv.ops.multi_scale_deform_attn import (MultiScaleDeformableAttention,
                                              MultiScaleDeformableAttnFunction,
                                              multi_scale_deformable_attn_pytorch)
from mmcv.runner.base_module import BaseModule
from mmdet.models.utils.transformer import (DeformableDetrTransformer,
                                            Transformer, inverse_sigmoid)

from .builder import (TRANSFORMER, ATTENTION, TRANSFORMER_LAYER_SEQUENCE,
                      build_transformer_layer_sequence)


@TRANSFORMER.register_module()
class SOITTransformer(DeformableDetrTransformer):
    """Implements the SOIT transformer.

    Args:
        mask_channels (int): Number of channels of output mask feature.
        seg_encoder (obj:`ConfigDict`): ConfigDict is used for building the
            encoder for mask feature generation.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(self,
                 mask_channels=8,
                 seg_encoder=dict(
                     type='DetrTransformerEncoder',
                     num_layers=1,
                     transformerlayers=dict(
                         type='BaseTransformerLayer',
                         attn_cfgs=dict(
                             type='MultiScaleDeformableAttention',
                             embed_dims=256,
                             num_heads=1,
                             num_levels=1),
                         feedforward_channels=1024,
                         ffn_dropout=0.1,
                         operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
                 as_two_stage=False,
                 num_feature_levels=4,
                 two_stage_num_proposals=300,
                 **kwargs):
        super(SOITTransformer, self).__init__(
            as_two_stage=as_two_stage, 
            num_feature_levels=num_feature_levels,
            two_stage_num_proposals=two_stage_num_proposals,
            **kwargs)
        self.seg_encoder = build_transformer_layer_sequence(seg_encoder)
        self.mask_channels = mask_channels
        self.mask_trans = nn.Linear(self.embed_dims, self.mask_channels)
        self.mask_trans_norm = nn.LayerNorm(self.mask_channels)
    
    def forward(self,
                mlvl_feats,
                mlvl_masks,
                query_embed,
                mlvl_pos_embeds,
                reg_branches=None,
                cls_branches=None,
                **kwargs):
        """Forward function for `Transformer`.

        Args:
            mlvl_feats (list(Tensor)): Input queries from
                different level. Each element has shape
                [bs, embed_dims, h, w].
            mlvl_masks (list(Tensor)): The key_padding_mask from
                different level used for encoder and decoder,
                each element has shape [bs, h, w].
            query_embed (Tensor): The query embedding for decoder,
                with shape [num_query, c].
            mlvl_pos_embeds (list(Tensor)): The positional encoding
                of feats from different level, has the shape
                [bs, embed_dims, h, w].
            reg_branches (obj:`nn.ModuleList`): Regression heads for
                feature maps from each decoder layer. Only would be
                passed when `with_box_refine` is Ture. Default to None.
            cls_branches (obj:`nn.ModuleList`): Classification heads
                for feature maps from each decoder layer. Only would
                be passed when `as_two_stage` is Ture. Default to None.

        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.

                - inter_states: Outputs from decoder. If
                    return_intermediate_dec is True output has shape \
                      (num_dec_layers, bs, num_query, embed_dims), else has \
                      shape (1, bs, num_query, embed_dims).
                - init_reference_out: The initial value of reference \
                    points, has shape (bs, num_queries, 4).
                - inter_references_out: The internal value of reference \
                    points in decoder, has shape \
                    (num_dec_layers, bs,num_query, embed_dims)
                - enc_outputs_class: The classification score of \
                    proposals generated from \
                    encoder's feature maps, has shape \
                    (batch, h*w, num_classes). \
                    Only would be returned when `as_two_stage` is True, \
                    otherwise None.
                - enc_outputs_coord_unact: The regression results \
                    generated from encoder's feature maps., has shape \
                    (batch, h*w, 4). Only would \
                    be returned when `as_two_stage` is True, \
                    otherwise None.
                - mask_proto: Feature, positional encoding and other \
                    information for mask feature.
        """
        assert self.as_two_stage or query_embed is not None

        feat_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(
                zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            feat = feat.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            feat_flatten.append(feat)
            mask_flatten.append(mask)
        feat_flatten = torch.cat(feat_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_flatten.device)
            
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1, )), spatial_shapes.prod(1).cumsum(0)[:-1])) 
        valid_ratios = torch.stack(
            [self.get_valid_ratio(m) for m in mlvl_masks], 1)
        reference_points = \
            self.get_reference_points(spatial_shapes,
                                      valid_ratios,
                                      device=feat.device) 

        feat_flatten = feat_flatten.permute(1, 0, 2)  # (H*W, bs, embed_dims)
        lvl_pos_embed_flatten = lvl_pos_embed_flatten.permute(
            1, 0, 2)  # (H*W, bs, embed_dims)
        memory = self.encoder(
            query=feat_flatten,
            key=None,
            value=None,
            query_pos=lvl_pos_embed_flatten,
            query_key_padding_mask=mask_flatten,
            spatial_shapes=spatial_shapes,
            reference_points=reference_points,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            **kwargs)

        memory = memory.permute(1, 0, 2)
        bs, _, c = memory.shape
        seg_memory = memory[:, level_start_index[0]:level_start_index[1], :]
        seg_pos_embed = lvl_pos_embed_flatten[
            level_start_index[0]:level_start_index[1], :, :]
        seg_mask = mask_flatten[:, level_start_index[0]:level_start_index[1]]
        seg_reference_points = reference_points[
            :, level_start_index[0]:level_start_index[1], [0], :]
        seg_memory = seg_memory.permute(1, 0, 2)

        seg_memory = self.seg_encoder(
            query=seg_memory,
            key=None,
            value=None,
            query_pos=seg_pos_embed,
            query_key_padding_mask=seg_mask,
            spatial_shapes=spatial_shapes[[0]],
            reference_points=seg_reference_points,
            level_start_index=level_start_index[0],
            valid_ratios=valid_ratios[:, [0], :],
            **kwargs)
        
        seg_memory = self.mask_trans_norm(self.mask_trans(seg_memory))
        mask_proto = (seg_memory, seg_pos_embed, seg_mask,
                      spatial_shapes[[0]], seg_reference_points,
                      level_start_index[0], valid_ratios[:, [0], :])
        
        if self.as_two_stage:
            output_memory, output_proposals = \
                self.gen_encoder_output_proposals(
                    memory, mask_flatten, spatial_shapes)
            enc_outputs_class = cls_branches[self.decoder.num_layers](
                output_memory)
            enc_outputs_coord_unact = \
                reg_branches[
                    self.decoder.num_layers](output_memory) + output_proposals

            topk = self.two_stage_num_proposals
            topk_proposals = torch.topk(
                enc_outputs_class[..., 0], topk, dim=1)[1]
            topk_coords_unact = torch.gather(
                enc_outputs_coord_unact, 1,
                topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid()
            init_reference_out = reference_points
            pos_trans_out = self.pos_trans_norm(
                self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))
            query_pos, query = torch.split(pos_trans_out, c, dim=2)
        else:
            query_pos, query = torch.split(query_embed, c, dim=1)
            query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
            query = query.unsqueeze(0).expand(bs, -1, -1)
            reference_points = self.reference_points(query_pos).sigmoid()
            init_reference_out = reference_points

        # decoder
        query = query.permute(1, 0, 2)
        memory = memory.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=memory,
            query_pos=query_pos,
            key_padding_mask=mask_flatten,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=reg_branches,
            **kwargs)

        inter_references_out = inter_references
        if self.as_two_stage:
            return inter_states, init_reference_out,\
                inter_references_out, enc_outputs_class,\
                enc_outputs_coord_unact, mask_proto
        return inter_states, init_reference_out, \
            inter_references_out, None, None, mask_proto


@ATTENTION.register_module()
class MultiScaleDeformablePoseAttention(BaseModule):
    """An attention module used in PETR. `End-to-End Multi-Person
    Pose Estimation with Transformers`.

    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 8.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 17.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_residual`.
            Default: 0.1.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=17,
                 im2col_step=64,
                 dropout=0.1,
                 norm_cfg=None,
                 init_cfg=None,
                 batch_first=False):
        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.init_cfg = init_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims,
                                           num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0.)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)

    def forward(self,
                query,
                key,
                value,
                residual=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):
        """Forward Function of MultiScaleDeformAttention.

        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape (num_key, bs, embed_dims).
            value (Tensor): The value tensor with shape
                (num_key, bs, embed_dims).
            residual (Tensor): The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            reference_points (Tensor):  The normalized reference points with
                shape (bs, num_query, num_levels, K*2), all elements is range
                in [0, 1], top-left (0,0), bottom-right (1, 1), including
                padding area.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different level. With shape  (num_levels, 2),
                last dimension represent (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].

        Returns:
            Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if key is None:
            key = query
        if value is None:
            value = key

        if residual is None:
            inp_residual = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_key, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_key

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_key, self.num_heads, -1)
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)
        if reference_points.shape[-1] == self.num_points * 2:
            reference_points_reshape = reference_points.reshape(
                bs, num_query, self.num_levels, -1, 2).unsqueeze(2)
            x1 = reference_points[:, :, :, 0::2].min(dim=-1, keepdim=True)[0]
            y1 = reference_points[:, :, :, 1::2].min(dim=-1, keepdim=True)[0]
            x2 = reference_points[:, :, :, 0::2].max(dim=-1, keepdim=True)[0]
            y2 = reference_points[:, :, :, 1::2].max(dim=-1, keepdim=True)[0]
            w = torch.clamp(x2 - x1, min=1e-4)
            h = torch.clamp(y2 - y1, min=1e-4)
            wh = torch.cat([w, h], dim=-1)[:, :, None, :, None, :]

            sampling_locations = reference_points_reshape \
                                 + sampling_offsets * wh * 0.5
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2K, but get {reference_points.shape[-1]} instead.')
        if torch.cuda.is_available():
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations,
                attention_weights)
        output = self.output_proj(output).permute(1, 0, 2)
        # (num_query, bs ,embed_dims)
        return self.dropout(output) + inp_residual


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class PetrTransformerDecoder(TransformerLayerSequence):
    """Implements the decoder in PETR transformer.

    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self,
                 *args,
                 return_intermediate=False,
                 num_keypoints=17,
                 **kwargs):

        super(PetrTransformerDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.num_keypoints = num_keypoints

    def forward(self,
                query,
                *args,
                reference_points=None,
                valid_ratios=None,
                kpt_branches=None,
                **kwargs):
        """Forward function for `TransformerDecoder`.

        Args:
            query (Tensor): Input query with shape (num_query, bs, embed_dims).
            reference_points (Tensor): The reference points of offset,
                has shape (bs, num_query, K*2).
            valid_ratios (Tensor): The radios of valid points on the feature
                map, has shape (bs, num_levels, 2).
            kpt_branches: (obj:`nn.ModuleList`): Used for refining the
                regression results. Only would be passed when `with_box_refine`
                is True, otherwise would be passed a `None`.

        Returns:
            tuple (Tensor): Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims] and
                [num_layers, bs, num_query, K*2].
        """
        output = query
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == self.num_keypoints * 2:
                reference_points_input = \
                    reference_points[:, :, None] * \
                    valid_ratios.repeat(1, 1, self.num_keypoints)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * \
                                         valid_ratios[:, None]
            output = layer(
                output,
                *args,
                reference_points=reference_points_input,
                **kwargs)
            output = output.permute(1, 0, 2)

            if kpt_branches is not None:
                tmp = kpt_branches[lid](output)
                if reference_points.shape[-1] == self.num_keypoints * 2:
                    new_reference_points = tmp + inverse_sigmoid(
                        reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    raise NotImplementedError
                reference_points = new_reference_points.detach()

            output = output.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)

        return output, reference_points


@TRANSFORMER.register_module()
class PETRTransformer(Transformer):
    """Implements the PETR transformer.

    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(self,
                 hm_encoder=dict(
                     type='DetrTransformerEncoder',
                     num_layers=1,
                     transformerlayers=dict(
                         type='BaseTransformerLayer',
                         attn_cfgs=dict(
                             type='MultiScaleDeformableAttention',
                             embed_dims=256,
                             num_levels=1),
                         feedforward_channels=1024,
                         ffn_dropout=0.1,
                         operation_order=('self_attn', 'norm', 'ffn',
                                          'norm'))),
                 refine_decoder=dict(
                     type='DeformableDetrTransformerDecoder',
                     num_layers=1,
                     return_intermediate=True,
                     transformerlayers=dict(
                         type='DetrTransformerDecoderLayer',
                         attn_cfgs=[
                             dict(
                                 type='MultiheadAttention',
                                 embed_dims=256,
                                 num_heads=8,
                                 dropout=0.1),
                             dict(
                                 type='MultiScaleDeformableAttention',
                                 embed_dims=256)
                         ],
                         feedforward_channels=1024,
                         ffn_dropout=0.1,
                         operation_order=('self_attn', 'norm', 'cross_attn',
                                          'norm', 'ffn', 'norm'))),
                 as_two_stage=True,
                 num_feature_levels=4,
                 two_stage_num_proposals=300,
                 num_keypoints=17,
                 **kwargs):
        super(PETRTransformer, self).__init__(**kwargs)
        self.as_two_stage = as_two_stage
        self.num_feature_levels = num_feature_levels
        self.two_stage_num_proposals = two_stage_num_proposals
        self.embed_dims = self.encoder.embed_dims
        self.num_keypoints = num_keypoints
        self.init_layers()
        self.hm_encoder = build_transformer_layer_sequence(hm_encoder)
        self.refine_decoder = build_transformer_layer_sequence(refine_decoder)

    def init_layers(self):
        """Initialize layers of the DeformableDetrTransformer."""
        self.level_embeds = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))

        if self.as_two_stage:
            self.enc_output = nn.Linear(self.embed_dims, self.embed_dims)
            self.enc_output_norm = nn.LayerNorm(self.embed_dims)
            self.refine_query_embedding = nn.Embedding(self.num_keypoints,
                                                       self.embed_dims * 2)
        else:
            self.reference_points = nn.Linear(self.embed_dims,
                                              2 * self.num_keypoints)

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()
        for m in self.modules():
            if isinstance(m, MultiScaleDeformablePoseAttention):
                m.init_weights()
        if not self.as_two_stage:
            xavier_init(self.reference_points, distribution='uniform', bias=0.)
        normal_(self.level_embeds)
        normal_(self.refine_query_embedding.weight)

    def gen_encoder_output_proposals(self, memory, memory_padding_mask,
                                     spatial_shapes):
        """Generate proposals from encoded memory.

        Args:
            memory (Tensor): The output of encoder, has shape
                (bs, num_key, embed_dim). num_key is equal the number of points
                on feature map from all level.
            memory_padding_mask (Tensor): Padding mask for memory.
                has shape (bs, num_key).
            spatial_shapes (Tensor): The shape of all feature maps.
                has shape (num_level, 2).

        Returns:
            tuple: A tuple of feature map and bbox prediction.

                - output_memory (Tensor): The input of decoder, has shape
                    (bs, num_key, embed_dim). num_key is equal the number of
                    points on feature map from all levels.
                - output_proposals (Tensor): The normalized proposal
                    after a inverse sigmoid, has shape (bs, num_keys, 4).
        """

        N, S, C = memory.shape
        proposals = []
        _cur = 0
        for lvl, (H, W) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H * W)].view(
                N, H, W, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(
                torch.linspace(
                    0, H - 1, H, dtype=torch.float32, device=memory.device),
                torch.linspace(
                    0, W - 1, W, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1),
                               valid_H.unsqueeze(-1)], 1).view(N, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N, -1, -1, -1) + 0.5) / scale
            proposal = grid.view(N, -1, 2)
            proposals.append(proposal)
            _cur += (H * W)
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) &
                                  (output_proposals < 0.99)).all(
                                      -1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(
            memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(
            ~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = output_memory.masked_fill(
            memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid,
                                                  float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        """Get the reference points used in decoder.

        Args:
            spatial_shapes (Tensor): The shape of all feature maps,
                has shape (num_level, 2).
            valid_ratios (Tensor): The radios of valid points on the
                feature map, has shape (bs, num_levels, 2).
            device (obj:`device`): The device where reference_points should be.

        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """
        reference_points_list = []
        for lvl, (H, W) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=torch.float32, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=torch.float32, device=device))
                # ref_y, ref_x 모두 (H, W) 형상이며, 
                # ref_y = [[0.5, ... 0.5], [1.5, .., 1.5], ... [H-0.5, ... H-0.5]]
                # ref_x = [[0.5, ..., W-0.5], ..., [0.5, ..., W-0.5]]
            ref_y = ref_y.reshape(-1)[None] / (
                valid_ratios[:, None, lvl, 1] * H)
                # ref_y.reshape(-1)[None] => torch.Size([1, 14706])
                # [[0.5, 0.5, ..., H-0.5, H-0.5]]
            ref_x = ref_x.reshape(-1)[None] / (
                valid_ratios[:, None, lvl, 0] * W)
                # ref_x.reshape(-1)[None] => torch.Size([1, 14706])
                # [[0.5, ..., W-0.5, ..., 0.5, ..., W-0.5]]v
            ref = torch.stack((ref_x, ref_y), -1)
                # torch.Size([1, 14706, 2])
                # (valid_ratios 안 나눴다고 하면,)
            """
            tensor([[[  0.5000,   0.5000],
                     [  1.5000,   0.5000],
                     [  2.5000,   0.5000],
                     ...,
                     [   W-0.5,    H-0.5]]])
            """
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1) # torch.Size([1, sum(HW), 2])
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
            # reference_points[:, :, None] => torch.Size([1, sum(HW), 1, 2])
            # valid_ratios[:, None] => torch.Size([bs, 1, num_levels, 2])
            # ==> torch.Size([bs, sum(HW), num_levels, 2])        
        return reference_points

    def get_valid_ratio(self, mask):
        """Get the valid radios of feature maps of all level."""
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def get_proposal_pos_embed(self,
                               proposals,
                               num_pos_feats=128,
                               temperature=10000):
        """Get the position embedding of proposal."""
        scale = 2 * math.pi
        dim_t = torch.arange(
            num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature**(2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()),
                          dim=4).flatten(2)
        return pos

    def forward(self,
                mlvl_feats,
                mlvl_masks,
                query_embed,
                mlvl_pos_embeds,
                kpt_branches=None,
                cls_branches=None,
                **kwargs):
        """Forward function for `Transformer`.

        Args:
            mlvl_feats (list(Tensor)): Input queries from different level.
                Each element has shape [bs, embed_dims, h, w].
            mlvl_masks (list(Tensor)): The key_padding_mask from different
                level used for encoder and decoder, each element has shape
                [bs, h, w].
            query_embed (Tensor): The query embedding for decoder,
                with shape [num_query, c].
            mlvl_pos_embeds (list(Tensor)): The positional encoding
                of feats from different level, has the shape
                 [bs, embed_dims, h, w].
            kpt_branches (obj:`nn.ModuleList`): Keypoint Regression heads for
                feature maps from each decoder layer. Only would be passed when
                `with_box_refine` is Ture. Default to None.
            cls_branches (obj:`nn.ModuleList`): Classification heads for
                feature maps from each decoder layer. Only would be passed when
                `as_two_stage` is Ture. Default to None.

        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.

                - inter_states: Outputs from decoder. If
                    `return_intermediate_dec` is True output has shape \
                    (num_dec_layers, bs, num_query, embed_dims), else has \
                    shape (1, bs, num_query, embed_dims).
                - init_reference_out: The initial value of reference \
                    points, has shape (bs, num_queries, 4).
                - inter_references_out: The internal value of reference \
                    points in decoder, has shape \
                    (num_dec_layers, bs,num_query, embed_dims)
                - enc_outputs_class: The classification score of proposals \
                    generated from encoder's feature maps, has shape \
                    (batch, h*w, num_classes). \
                    Only would be returned when `as_two_stage` is True, \
                    otherwise None.
                - enc_outputs_kpt_unact: The regression results generated from \
                    encoder's feature maps., has shape (batch, h*w, K*2).
                    Only would be returned when `as_two_stage` is True, \
                    otherwise None.
        """
        assert self.as_two_stage or query_embed is not None

        feat_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(
                zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)):
            bs, c, h, w = feat.shape
            print(f"Feature map Level {lvl}. =====\n(bs, embed_dims, h, w) = ({bs}, {c}, {h}, {w})")
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            feat = feat.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            feat_flatten.append(feat)
            mask_flatten.append(mask)
            """
            Feature map Level 0. =====
            (bs, embed_dims, h, w) = (1, 256, 100, 150)
            Feature map Level 1. =====
            (bs, embed_dims, h, w) = (1, 256, 50, 75)
            Feature map Level 2. =====
            (bs, embed_dims, h, w) = (1, 256, 25, 38)
            Feature map Level 3. =====
            (bs, embed_dims, h, w) = (1, 256, 13, 19)
            """
        feat_flatten = torch.cat(feat_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_flatten.device)
            # ex) [[114, 129], [57, 65], [29, 33], [15, 17]] => (num_levels, 2) => 각 피처맵의 사이즈
        print(f"spatial_shapes (shape: {spatial_shapes.shape}) =====\n{spatial_shapes}")
        # spatial_shapes (shape: torch.Size([4, 2]))

        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
            # torch.Size([4])
            # ex) tensor([    0., 14706., 18411., 19368.]) => feat_flatten에서 각 피처맵이 어디서부터 시작하는지
        print(f"level_start_index (shape: {level_start_index.shape}) =====\n{level_start_index}")
        """ level_start_index (shape: torch.Size([4])) =====
            tensor([    0, 15000, 18750, 19700])"""

        valid_ratios = torch.stack(
            [self.get_valid_ratio(m) for m in mlvl_masks], 1)
            # (bs, num_levels, 2)
            # The ratios of valid points on the feature map
        print(f"valid_ratios (shape: {valid_ratios.shape}) =====\n{valid_ratios}")
        """valid_ratios (shape: torch.Size([1, 4, 2])) =====
            tensor([[[1., 1.],
                    [1., 1.],
                    [1., 1.],
                    [1., 1.]]])"""

        reference_points = \
            self.get_reference_points(spatial_shapes,
                                      valid_ratios,
                                      device=feat.device)
            #torch.Size([bs, sum(HW), num_levels, 2])
        print(f"reference_points (shape: {reference_points.shape})")
        # reference_points (shape: torch.Size([1, 19947, 4, 2]))

        feat_flatten = feat_flatten.permute(1, 0, 2)  # (sum(hw), bs, embed_dims)
        print(f"feat_flatten (shape: {feat_flatten.shape})")
        # feat_flatten (shape: torch.Size([19947, 1, 256]))

        lvl_pos_embed_flatten = lvl_pos_embed_flatten.permute(
            1, 0, 2)  # (sum(hw), bs, embed_dims)
        print(f"lvl_pos_embed_flatten (shape: {lvl_pos_embed_flatten.shape})")
        # lvl_pos_embed_flatten (shape: torch.Size([19947, 1, 256]))

        memory = self.encoder(
            query=feat_flatten,
            key=None,
            value=None,
            query_pos=lvl_pos_embed_flatten,
            query_key_padding_mask=mask_flatten,
            spatial_shapes=spatial_shapes,
            reference_points=reference_points,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            **kwargs)

        memory = memory.permute(1, 0, 2)
        print(f"memory(result of encoder) (shape: {memory.shape})")
        # memory(result of encoder) (shape: torch.Size([1, 19947, 256]))
        bs, _, c = memory.shape
            # (bs, num_key, embed_dim). 
            # "num_key" = number of points on feature map from all level

        hm_proto = None
        if self.training:
            hm_memory = memory[
                :, level_start_index[0]:level_start_index[1], :]
            hm_pos_embed = lvl_pos_embed_flatten[
                level_start_index[0]:level_start_index[1], :, :]
            hm_mask = mask_flatten[
                :, level_start_index[0]:level_start_index[1]]
            hm_reference_points = reference_points[
                :, level_start_index[0]:level_start_index[1], [0], :]
            hm_memory = hm_memory.permute(1, 0, 2)
            hm_memory = self.hm_encoder(
                query=hm_memory,
                key=None,
                value=None,
                query_pose=hm_pos_embed,
                query_key_padding_mask=hm_mask,
                spatial_shapes=spatial_shapes[[0]],
                reference_points=hm_reference_points,
                level_start_index=level_start_index[0],
                valid_ratios=valid_ratios[:, [0], :],
                **kwargs)
            hm_memory = hm_memory.permute(1, 0, 2).reshape(bs,
                spatial_shapes[0, 0], spatial_shapes[0, 1], -1)
            hm_proto = (hm_memory, mlvl_masks[0])

        if self.as_two_stage:
            output_memory, output_proposals = \
                self.gen_encoder_output_proposals(
                    memory, mask_flatten, spatial_shapes)
                # output_memory (Tensor) => (bs, num_key(=전체 레벨 포인트 개수), embed_dim)
                #   디코더의 입력으로 쓸 피처맵
                # output_proposals (Tensor) => (bs, num_keys, 2)
                #   역시그모이드 거쳐 정규화한 bbox proposal
            print(f"output_memory (shape: {output_memory.shape})")
            # output_memory (shape: torch.Size([1, 19947, 256]))
            print(f"output_proposals (shape: {output_proposals.shape})")
            # output_proposals (shape: torch.Size([1, 19947, 2]))

            enc_outputs_class = cls_branches[self.decoder.num_layers](
                output_memory)
                # petr_head.py에서 확인! cls_branches: fc_cls라는 Linear 모듈(embed_dims, cls_out_channels(=1))이 num_pred(=디코더 레이어 수 + 1) 만큼 있음
                # (bs, num_key(=전체 레벨 포인트 개수), cls_out_channels(=1))
            print(f"enc_outputs_class (shape: {enc_outputs_class.shape})")
            # enc_outputs_class (shape: torch.Size([1, 19947, 1]))

            enc_outputs_kpt_unact = \
                kpt_branches[self.decoder.num_layers](output_memory)
                # petr_head.py에서 확인! kpt_branches: kpt_branches라는 Sequential 모듈(embed_dims=>2*num_keypoints(=2*17))이 num_pred(=디코더 레이어 수 + 1) 만큼 있음
                # (bs, num_key(=전체 레벨 포인트 개수), 2*num_keypoints(=2*17))
            print(f"enc_outputs_kpt_unact (shape: {enc_outputs_kpt_unact.shape})")
            # enc_outputs_kpt_unact (shape: torch.Size([1, 19947, 34]))

            enc_outputs_kpt_unact[..., 0::2] += output_proposals[..., 0:1] # TODO shape: (bs, num_keys, 1) -> x좌표
            enc_outputs_kpt_unact[..., 1::2] += output_proposals[..., 1:2] # TODO shape: (bs, num_keys, 1) -> y좌표
            # TODO 왜 더하는 거지?? 무슨 의미지??

            topk = self.two_stage_num_proposals # 300
            print(f"topk = {topk}")

            topk_proposals = torch.topk(
                enc_outputs_class[..., 0], topk, dim=1)[1]
                # torch.topk(inputTensor, k): top-k 개를 선정 => 결과 튜플 (높은 순대로 오름차순한 값의 리스트, 그들의 원래 인덱스)
                # torch.topk()[1] => 높은 값부터 enc_outputs_class[..., 0]에서의 인덱스 => torch.Size((bs, topk))
            print(f"topk_proposals (shape: {topk_proposals.shape})")
            # topk_proposals (shape: torch.Size([1, 300]))

            # XXX 원래 코드의 주석임
            # topk_coords_unact = torch.gather(
            #     enc_outputs_coord_unact, 1,
            #     topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            # topk_coords_unact = topk_coords_unact.detach()

            topk_kpts_unact = torch.gather(
                enc_outputs_kpt_unact, 1,
                topk_proposals.unsqueeze(-1).repeat(
                    1, 1, enc_outputs_kpt_unact.size(-1)))
                    # topk_proposals.unsqueeze(-1): (bs, topk, 1)
                    # topk_proposals.unsqueeze(-1).repeat(): (bs, topk, 2*num_keypoints)
                    # ==> topk_kpts_unact: (bs, topk, 2*num_keypoints)
            topk_kpts_unact = topk_kpts_unact.detach()
            print(f"topk_kpts_unact (shape: {topk_kpts_unact.shape})")
            # (bs, topk, 2*17)
            # topk_kpts_unact (shape: torch.Size([1, 300, 34]))

            reference_points = topk_kpts_unact.sigmoid() # (bs, topk, 2*num_keypoints) # TODO 여기서부터 분석 다시
            init_reference_out = reference_points
            print(f"reference_points, init_reference_out (shape: {init_reference_out.shape})")
            # (bs, topk, 2*17)
            # reference_points, init_reference_out (shape: torch.Size([1, 300, 34]))

            # learnable query and query_pos
            query_pos, query = torch.split(query_embed, c, dim=1) # TODO query_embed의 (num_qeury, c)의 c와 memory의 형상 중 c가 같다면 이 결과는 1개의 텐서(torch.Size([num_query, c]))만 나옴
            query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
            query = query.unsqueeze(0).expand(bs, -1, -1)
        else:
            query_pos, query = torch.split(query_embed, c, dim=1)
            query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
            query = query.unsqueeze(0).expand(bs, -1, -1)
            reference_points = self.reference_points(query_pos).sigmoid()
            init_reference_out = reference_points

        # decoder
        query = query.permute(1, 0, 2)
        print(f"query (shape: {query.shape})")
        # (topk, bs, dim)
        # query (shape: torch.Size([300, 1, 256]))

        memory = memory.permute(1, 0, 2)
        print(f"memory(permute 'output of encoder' & give as input to decoder)\n  (shape: {memory.shape})")
        # (sum(hw), bs, dim)
        # (shape: torch.Size([19947, 1, 256]))
        
        query_pos = query_pos.permute(1, 0, 2)
        print(f"query_pos (shape: {query_pos.shape})")
        # (topk, bs, dim)
        # query_pos (shape: torch.Size([300, 1, 256]))

        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=memory,
            query_pos=query_pos,
            key_padding_mask=mask_flatten,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            kpt_branches=kpt_branches,
            **kwargs)
        print(f"inter_states (shape: {inter_states.shape})")
        # (num_decoder_layers, topk, bs, embed_dims)
        # inter_states (shape: torch.Size([3, 300, 1, 256]))
        print(f"inter_references (shape: {inter_references.shape})")
        # # (num_decoder_layers, bs, topk, 2*17)
        # inter_references (shape: torch.Size([3, 1, 300, 34]))

        inter_references_out = inter_references
        if self.as_two_stage:
            return inter_states, init_reference_out, \
                   inter_references_out, enc_outputs_class, \
                   enc_outputs_kpt_unact, hm_proto, memory
        return inter_states, init_reference_out, \
               inter_references_out, None, None, None, None, None, hm_proto

    def forward_refine(self,
                       mlvl_masks,
                       memory,
                       reference_points_pose,
                       img_inds,
                       kpt_branches=None,
                       **kwargs):
        mask_flatten = []
        spatial_shapes = []
        for lvl, mask in enumerate(mlvl_masks):
            bs, h, w = mask.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            mask = mask.flatten(1)
            mask_flatten.append(mask)
        mask_flatten = torch.cat(mask_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=mask_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack(
            [self.get_valid_ratio(m) for m in mlvl_masks], 1)

        # pose refinement (17 queries corresponding to 17 keypoints)
        # learnable query and query_pos
        refine_query_embedding = self.refine_query_embedding.weight
        query_pos, query = torch.split(
            refine_query_embedding, refine_query_embedding.size(1) // 2, dim=1)
        pos_num = reference_points_pose.size(0)
        query_pos = query_pos.unsqueeze(0).expand(pos_num, -1, -1)
        query = query.unsqueeze(0).expand(pos_num, -1, -1)
        reference_points = reference_points_pose.reshape(
            pos_num,
            reference_points_pose.size(1) // 2, 2)
        query = query.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        pos_memory = memory[:, img_inds, :]
        mask_flatten = mask_flatten[img_inds, :]
        valid_ratios = valid_ratios[img_inds, ...]
        inter_states, inter_references = self.refine_decoder(
            query=query,
            key=None,
            value=pos_memory,
            query_pos=query_pos,
            key_padding_mask=mask_flatten,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=kpt_branches,
            **kwargs)
        # [num_decoder, num_query, bs, embed_dim]

        init_reference_out = reference_points
        return inter_states, init_reference_out, inter_references
