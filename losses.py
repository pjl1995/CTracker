import numpy as np
import torch
import torch.nn as nn
from utils import BasicBlock, Bottleneck, BBoxTransform, ClipBoxes
from anchors import Anchors

def calc_iou(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih

    ua = torch.clamp(ua, min=1e-8)

    intersection = iw * ih

    IoU = intersection / ua

    return IoU

class FocalLoss(nn.Module):
    def forward(self, classifications, regressions, anchors, annotations1, annotations2):
        alpha = 0.25
        gamma = 2.0
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []

        anchor = anchors[0, :, :]

        anchor_widths  = anchor[:, 2] - anchor[:, 0]
        anchor_heights = anchor[:, 3] - anchor[:, 1]
        anchor_ctr_x   = anchor[:, 0] + 0.5 * anchor_widths
        anchor_ctr_y   = anchor[:, 1] + 0.5 * anchor_heights

        for j in range(batch_size):
            # get gt box
            classification = classifications[j, :, :]
            regression = regressions[j, :, :]

            bbox_annotation = annotations1[j, :, :]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]

            bbox_annotation_next = annotations2[j, :, :]
            bbox_annotation_next = bbox_annotation_next[bbox_annotation_next[:, 4] != -1]

            # return zero loss if no gt boxes exist
            if bbox_annotation.shape[0] == 0:
                regression_losses.append(torch.tensor(0).float().cuda())
                classification_losses.append(torch.tensor(0).float().cuda())
                continue

            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)
            # IoU matching for latter anchors' label assign
            IoU = calc_iou(anchors[0, :, :], bbox_annotation[:, :4]) # num_anchors x num_annotations
            best_truth_overlap, best_truth_idx = torch.max(IoU, dim=1) # num_anchors x 1

            best_prior_overlap, best_prior_idx = torch.max(IoU, dim=0) # num_annotations
            best_truth_overlap.index_fill_(0, best_prior_idx, 2)  # ensure best prior
            for j in range(best_prior_idx.size(0)):
                best_truth_idx[best_prior_idx[j]] = j

            # compute the label for classification
            targets = torch.ones(classification.shape) * -1
            targets = targets.cuda()

            targets[torch.lt(best_truth_overlap, 0.4), :] = 0

            positive_indices = torch.ge(best_truth_overlap, 0.5)

            num_positive_anchors = positive_indices.sum()

            assigned_annotations = bbox_annotation[best_truth_idx, :]

            targets[positive_indices, :] = 0
            targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1

            # compute the loss for classification
            alpha_factor = torch.ones(targets.shape).cuda() * alpha

            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

            bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))

            cls_loss = focal_weight * bce

            cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).cuda())

            classification_losses.append(cls_loss.sum()/torch.clamp(num_positive_anchors.float(), min=1.0))

            # compute the loss for regression
            if positive_indices.sum() > 0:
                assigned_annotations = assigned_annotations[positive_indices, :]
                # assigned_annotations_next = assigned_annotations_next[positive_indices, :]
                assigned_ids = assigned_annotations[:, 5]
                assigned_annotations_next = torch.zeros_like(assigned_annotations)
                reg_mask = torch.ones(assigned_annotations.shape[0], 8).cuda()
                # only learn regression for chained-anchors, whose id of target in two frames are the same
                for m in range(assigned_annotations_next.shape[0]):
                    assigned_id = assigned_annotations[m, 5]
                    match_flag = False
                    for n in range(bbox_annotation_next.shape[0]):
                        if bbox_annotation_next[n, 5] == assigned_id:
                            match_flag = True
                            assigned_annotations_next[m, :] = bbox_annotation_next[n, :]
                            break
                    if match_flag == False:
                        reg_mask[m, 4:] = 0
    
                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]
                # transform GT to x,y,w,h
                gt_widths  = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                gt_ctr_x   = assigned_annotations[:, 0] + 0.5 * gt_widths
                gt_ctr_y   = assigned_annotations[:, 1] + 0.5 * gt_heights

                gt_widths_next  = assigned_annotations_next[:, 2] - assigned_annotations_next[:, 0]
                gt_heights_next = assigned_annotations_next[:, 3] - assigned_annotations_next[:, 1]
                gt_ctr_x_next   = assigned_annotations_next[:, 0] + 0.5 * gt_widths_next
                gt_ctr_y_next   = assigned_annotations_next[:, 1] + 0.5 * gt_heights_next

                # clip widths to 1
                gt_widths  = torch.clamp(gt_widths, min=1)
                gt_heights = torch.clamp(gt_heights, min=1)

                gt_widths_next  = torch.clamp(gt_widths_next, min=1)
                gt_heights_next = torch.clamp(gt_heights_next, min=1)
                # compute regression targets
                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw = torch.log(gt_widths / anchor_widths_pi)
                targets_dh = torch.log(gt_heights / anchor_heights_pi)

                targets_dx_next = (gt_ctr_x_next - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy_next = (gt_ctr_y_next - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw_next = torch.log(gt_widths_next / anchor_widths_pi)
                targets_dh_next = torch.log(gt_heights_next / anchor_heights_pi)


                targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh, targets_dx_next, targets_dy_next, targets_dw_next, targets_dh_next))
                targets = targets.t()

                targets = targets/torch.Tensor([[0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.2, 0.2]]).cuda()
                # compute losses
                regression_diff = torch.abs(targets - regression[positive_indices, :]) * reg_mask

                regression_loss = torch.where(
                    torch.le(regression_diff, 1.0 / 9.0),
                    0.5 * 9.0 * torch.pow(regression_diff, 2),
                    regression_diff - 0.5 / 9.0
                )

                regression_losses.append(regression_loss.mean())
            else:
                regression_losses.append(torch.tensor(0).float().cuda())

        return torch.stack(classification_losses).mean(dim=0, keepdim=True), torch.stack(regression_losses).mean(dim=0, keepdim=True)

    
class FocalLossReid(nn.Module):
    def forward(self, classifications, anchors, annotations1, annotations2):
        alpha = 0.25
        gamma = 2.0
        batch_size = classifications.shape[0]
        classification_losses = []

        for j in range(batch_size):
            # get gt box
            classification = classifications[j, :, :]

            bbox_annotation = annotations1[j, :, :]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]

            bbox_annotation_next = annotations2[j, :, :]
            bbox_annotation_next = bbox_annotation_next[bbox_annotation_next[:, 4] != -1]
            # return zero loss if no gt boxes exist
            if bbox_annotation.shape[0] == 0:
                classification_losses.append(torch.tensor(0).float().cuda())
                continue

            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)

            # compute the label for classification
            targets = torch.ones(classification.shape) * -1
            targets = targets.cuda()

            # IoU matching for latter anchors' label assign, current frame
            IoU = calc_iou(anchors[0, :, :], bbox_annotation[:, :4]) # num_anchors x num_annotations

            best_truth_overlap, best_truth_idx = torch.max(IoU, dim=1) # num_anchors x 1

            _, best_prior_idx = torch.max(IoU, dim=0) # num_annotations
            best_truth_overlap.index_fill_(0, best_prior_idx, 2)  # ensure best prior
            for j in range(best_prior_idx.size(0)):
                best_truth_idx[best_prior_idx[j]] = j
 
            assigned_annotations = bbox_annotation[best_truth_idx, :]
            # IoU matching for latter anchors' label assign, next frame
            IoU_next = calc_iou(anchors[0, :, :], bbox_annotation_next[:, :4]) # num_anchors x num_annotations
            best_truth_overlap_next, best_truth_idx_next = torch.max(IoU_next, dim=1) # num_anchors x 1

            _, best_prior_idx_next = torch.max(IoU_next, dim=0) # num_annotations
            best_truth_overlap_next.index_fill_(0, best_prior_idx_next, 2)  # ensure best prior
            for j in range(best_prior_idx_next.size(0)):
                best_truth_idx_next[best_prior_idx_next[j]] = j

            assigned_annotations_next = bbox_annotation_next[best_truth_idx_next, :]

            reid_pos_thres = 0.5#0.7

            # label for reid branch is 1 if and only if the following criterion is satisfied
            valid_samples = (torch.ge(best_truth_overlap, reid_pos_thres) & torch.ge(best_truth_overlap_next, 0.4)) | (torch.ge(best_truth_overlap_next, reid_pos_thres) & torch.ge(best_truth_overlap, 0.4))
            targets[valid_samples & torch.eq(assigned_annotations[:, 5], assigned_annotations_next[:, 5]), :] = 1
            targets[valid_samples & torch.ne(assigned_annotations[:, 5], assigned_annotations_next[:, 5]), :] = 0

            targets[torch.lt(best_truth_overlap, 0.4) | torch.lt(best_truth_overlap_next, 0.4), :] = 0
            targets[torch.lt(best_truth_overlap, reid_pos_thres) & torch.ge(best_truth_overlap, 0.4) & torch.lt(best_truth_overlap_next, reid_pos_thres) & torch.ge(best_truth_overlap_next, 0.4), :] = -1

            positive_indices = torch.ge(targets, 1)
            
            num_positive_anchors = positive_indices.sum()

            # compute losses
            alpha_factor = torch.ones(targets.shape).cuda() * alpha

            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

            bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))

            cls_loss = focal_weight * bce

            cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).cuda())

            classification_losses.append(cls_loss.sum()/torch.clamp(num_positive_anchors.float(), min=1.0))

        return torch.stack(classification_losses).mean(dim=0, keepdim=True)

    