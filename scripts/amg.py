# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import cv2  # type: ignore
import random
import os
os.sys.path.append("/home/industai/code_folder/Python_code/gitlab_project/segment-anything/")
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

import argparse
import json
from typing import Any, Dict, List
import torch
import numpy as np
import time
parser = argparse.ArgumentParser(
    description=(
        "Runs automatic mask generation on an input image or directory of images, "
        "and outputs masks as either PNGs or COCO-style RLEs. Requires open-cv, "
        "as well as pycocotools if saving in RLE format."
    )
)
#输入
parser.add_argument(
    "--input",
    type=str,
    required=True,
    help="Path to either a single input image or folder of images.",
)
#输出
parser.add_argument(
    "--output",
    type=str,
    required=True,
    help=(
        "Path to the directory where masks will be output. Output will be either a folder "
        "of PNGs per image or a single json with COCO-style masks."
    ),
)
#模型选择
parser.add_argument(
    "--model-type",
    type=str,
    required=True,
    help="The type of model to load, in ['default', 'vit_h', 'vit_l', 'vit_b']",
)
#模型权重
parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="The path to the SAM checkpoint to use for mask generation.",
)
#显卡选择
parser.add_argument("--device", type=str, default="cuda", help="The device to run generation on.")
#生成一个json文件
parser.add_argument(
    "--convert-to-rle",
    type=str,
    default="binary_mask",
    required=True,
    help=("binary_masks","mask_src","json")
)

amg_settings = parser.add_argument_group("AMG Settings")

#通过对网格进行采样生成蒙版
amg_settings.add_argument(
    "--points-per-side",
    type=int,
    default=None,
    help="Generate masks by sampling a grid over the image with this many points to a side.",
)
#输入几个点
amg_settings.add_argument(
    "--points-per-batch",
    type=int,
    default=None,
    help="How many input points to process simultaneously in one batch.",
)
#结果掩码阈值设置
amg_settings.add_argument(
    "--pred-iou-thresh",
    type=float,
    default=None,
    help="Exclude masks with a predicted score from the model that is lower than this threshold.",
)
#稳定性分数掩码阈值设置？
amg_settings.add_argument(
    "--stability-score-thresh",
    type=float,
    default=None,
    help="Exclude masks with a stability score lower than this threshold.",
)

#稳定性分数缩放因子？
amg_settings.add_argument(
    "--stability-score-offset",
    type=float,
    default=None,
    help="Larger values perturb the mask more when measuring stability score.",
)
#检测框的去重
amg_settings.add_argument(
    "--box-nms-thresh",
    type=float,
    default=None,
    help="The overlap threshold for excluding a duplicate mask.",
)
#不同的尺度进行分割
amg_settings.add_argument(
    "--crop-n-layers",
    type=int,
    default=None,
    help=(
        "If >0, mask generation is run on smaller crops of the image to generate more masks. "
        "The value sets how many different scales to crop at."
    ),
)
#掩码去重
amg_settings.add_argument(
    "--crop-nms-thresh",
    type=float,
    default=None,
    help="The overlap threshold for excluding duplicate masks across different crops.",
)
#分割范围扩张
amg_settings.add_argument(
    "--crop-overlap-ratio",
    type=int,
    default=None,
    help="Larger numbers mean image crops will overlap more.",
)

amg_settings.add_argument(
    "--crop-n-points-downscale-factor",
    type=int,
    default=None,
    help="The number of points-per-side in each layer of crop is reduced by this factor.",
)

#掩码区域截断
amg_settings.add_argument(
    "--min-mask-region-area",
    type=int,
    default=None,
    help=(
        "Disconnected mask regions or holes with area smaller than this value "
        "in pixels are removed by postprocessing."
    ),
)


def write_masks_to_folder(masks: List[Dict[str, Any]], path: str) -> None:
    header = "id,area,bbox_x0,bbox_y0,bbox_w,bbox_h,point_input_x,point_input_y,predicted_iou,stability_score,crop_box_x0,crop_box_y0,crop_box_w,crop_box_h"  # noqa
    metadata = [header]
    for i, mask_data in enumerate(masks):
        mask = mask_data["segmentation"]
        filename = f"{i}.png"
        cv2.imwrite(os.path.join(path, filename), mask * 255)
        mask_metadata = [
            str(i),
            str(mask_data["area"]),
            *[str(x) for x in mask_data["bbox"]],
            *[str(x) for x in mask_data["point_coords"][0]],
            str(mask_data["predicted_iou"]),
            str(mask_data["stability_score"]),
            *[str(x) for x in mask_data["crop_box"]],
        ]
        row = ",".join(mask_metadata)
        metadata.append(row)
    metadata_path = os.path.join(path, "metadata.csv")
    with open(metadata_path, "w") as f:
        f.write("\n".join(metadata))

    return

def mask_color_random(color_list,mask_len):
    
    _color=(random.randint(30,200),random.randint(30,200),random.randint(30,200))
    while (_color in color_list):
        _color=(random.randint(30,200),random.randint(30,200),random.randint(30,200))
    color_list.append(_color)
    if len(color_list)>=mask_len:
        return color_list
    else:
        color_list = mask_color_random(color_list,mask_len)
        return color_list


def mix_masks_src_img(img,masks: List[Dict[str, Any]], path: str) -> None:
    header = "id,area,bbox_x0,bbox_y0,bbox_w,bbox_h,point_input_x,point_input_y,predicted_iou,stability_score,crop_box_x0,crop_box_y0,crop_box_w,crop_box_h"  # noqa
    metadata = [header]
    color_list = []
    mask_len = len(masks)
    color_list = mask_color_random(color_list,mask_len)
    _mask = np.zeros(img.shape,dtype=np.uint8)
    for i, mask_data in enumerate(masks):
        mask = mask_data["segmentation"]
        mask_R = mask * color_list[i][0]
        mask_G = mask * color_list[i][1]
        mask_B = mask * color_list[i][2]
        mask_RGB = np.stack((mask_R,mask_G,mask_B),axis=2).astype(np.uint8)
        _mask = cv2.add(_mask,mask_RGB)
        # filename = f"{i}.png"
        # cv2.imwrite(os.path.join(path, filename),mask_RGB)
        mask_metadata = [
            str(i),
            str(mask_data["area"]),
            *[str(x) for x in mask_data["bbox"]],
            *[str(x) for x in mask_data["point_coords"][0]],
            str(mask_data["predicted_iou"]),
            str(mask_data["stability_score"]),
            *[str(x) for x in mask_data["crop_box"]],
        ]
        row = ",".join(mask_metadata)
        metadata.append(row)
    # cv2.imwrite(os.path.join(path, "mask.png"),_mask)
    mask_src = cv2.addWeighted(img,0.5,_mask,0.5,0)
    cv2.imwrite(os.path.join(path, "mask.png"),mask_src)

    metadata_path = os.path.join(path, "metadata.csv")
    with open(metadata_path, "w") as f:
        f.write("\n".join(metadata))

    return


def get_amg_kwargs(args):
    amg_kwargs = {
        "points_per_side": args.points_per_side,
        "points_per_batch": args.points_per_batch,
        "pred_iou_thresh": args.pred_iou_thresh,
        "stability_score_thresh": args.stability_score_thresh,
        "stability_score_offset": args.stability_score_offset,
        "box_nms_thresh": args.box_nms_thresh,
        "crop_n_layers": args.crop_n_layers,
        "crop_nms_thresh": args.crop_nms_thresh,
        "crop_overlap_ratio": args.crop_overlap_ratio,
        "crop_n_points_downscale_factor": args.crop_n_points_downscale_factor,
        "min_mask_region_area": args.min_mask_region_area,
    }
    amg_kwargs = {k: v for k, v in amg_kwargs.items() if v is not None}
    return amg_kwargs


def main(args: argparse.Namespace) -> None:
    print("Loading model...")
    device = torch.device(args.device)
    
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    device_ids = [0,1]
    sam = sam.cuda(device_ids[0])
    if torch.cuda.device_count() > 1:
        sam = torch.nn.DataParallel(sam,device_ids=device_ids)
    # _ = sam.to(device)
    output_mode = args.convert_to_rle
    amg_kwargs = get_amg_kwargs(args)
    generator = SamAutomaticMaskGenerator(sam, output_mode=output_mode, **amg_kwargs)

    if not os.path.isdir(args.input):
        targets = [args.input]
    else:
        targets = [
            f for f in os.listdir(args.input) if not os.path.isdir(os.path.join(args.input, f))
        ]
        targets = [os.path.join(args.input, f) for f in targets]

    os.makedirs(args.output, exist_ok=True)

    time_log=[]
    for t in targets:
        print(f"Processing '{t}'...")
        src_image = cv2.imread(t)
        if src_image is None:
            print(f"Could not load '{t}' as an image, skipping...")
            continue
        image = cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB)
        t1 = time.time()
        masks = generator.generate(image)
        t2 = time.time()
        time_log.append([(t2-t1),image.shape[0]*image.shape[1],len(masks)])
        print("平均推理耗时：{}".format(time_log[-1]))
        # base = os.path.basename(t)
        # base = os.path.splitext(base)[0]
        # save_base = os.path.join(args.output, base)
        # if output_mode == "binary_mask":
        #     os.makedirs(save_base, exist_ok=True)
        #     write_masks_to_folder(masks, save_base)
        # elif output_mode == "mask_src":
        #     os.makedirs(save_base, exist_ok=True)
        #     mix_masks_src_img(src_image,masks, save_base)
        # else:
        #     save_file = save_base + ".json"
        #     with open(save_file, "w") as f:
        #         json.dump(masks, f)
    time_array = np.array(time_log)
    np.savetxt("GPU_time.txt", time_array)
    avg_time = np.mean(time_array[:,0])
    avg_size = np.mean(time_array[:,1])
    avg_masks = np.mean(time_array[:,2])
    print("平均推理耗时：{},平均尺寸:{},平均mask数量:{}".format(avg_time,avg_size,avg_masks))
    print("Done!")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
