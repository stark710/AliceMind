import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
from PIL import Image
from PIL import ImageDraw
import torch
import torch.nn as nn
# import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import utils as public_utils
# from torch.utils.data import DataLoader
from dataset.grounding_dataset import NestedTensor, collate_fn, collate_fn_val, read_examples, convert_examples_to_features
import matplotlib.pyplot as plt
from models.model_grounding_mplug import MPLUG
# from models.vit import interpolate_pos_embed, resize_pos_embed
from models.vit import resize_pos_embed
from models.tokenization_bert import BertTokenizer

from vgTools.utils import misc as utils
# from dataset.utils import save_result
from dataset import create_dataset, create_sampler, create_loader
from scheduler import create_scheduler
from optim import create_optimizer, create_two_optimizer
from vgTools.utils import eval_utils
# from icecream import ic
# from pdb import set_trace as breakpoint
import torchvision.transforms as T
from dataset import vg_transforms as vgT 
from torchvision.utils import draw_bounding_boxes

def load_checkpoint(model,checkpoint_path,args,config):
    if isinstance(model,torch.nn.parallel.DistributedDataParallel):
        model=model.module
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model']
    tmp = {}
    for key in state_dict.keys():
        if '_m.' in key:
            continue
        if 'text_encoder.bert' in key[:len('text_encoder.bert')]:
            encoder_key = key.replace('bert.', '')
            tmp[encoder_key] = state_dict[key]
        elif 'fusion_encoder.fusion' in key:
            encoder_key = key.replace('fusion.', '')
            tmp[encoder_key]=state_dict[key]
        else:
            tmp[key]=state_dict[key]

    state_dict = tmp

    # reshape positional embedding to accomodate for image resolution change
    vit_rate = 16*16 if '16' in config['clip_name'] else 14*14
    num_patches = int(config["image_res"] * config["image_res"]/vit_rate)
    pos_embed = nn.Parameter(torch.zeros(num_patches + 1, config['vision_width']).float())

    pos_embed = resize_pos_embed(state_dict['visual_encoder.visual.positional_embedding'].unsqueeze(0),
                                                pos_embed.unsqueeze(0))
    state_dict['visual_encoder.visual.positional_embedding'] = pos_embed

    if not args.evaluate:
        if config['distill']:
            num_patches = int(config["image_res"] * config["image_res"] / vit_rate)
            pos_embed = nn.Parameter(torch.zeros(num_patches + 1, config['vision_width']).float())

    msg = model.load_state_dict(state_dict, strict=False)
    print('load checkpoint from %s' % checkpoint_path)
    print(msg)

# @torch.no_grad()
def evaluate(model, device, tokenizer, idx=0):
    imsize = 336
    model.eval()
    pred_box_list = []
    gt_box_list = []
    phrase = 'man in red shirt riding bike'
    phrase = phrase.lower()
    pil_img = Image.open('./test_images/man_riding_bike.jpg').convert("RGB")
    # img = torch.stack(img)
    input_dict = {}
    input_dict['img'] = pil_img
    input_dict['text'] = phrase
    input_dict['box'] = torch.FloatTensor([0,0,0,0])
    transforms = vgT.Compose([
        vgT.RandomResize([336]),
        vgT.ToTensor(),
        vgT.NormalizeAndPad(size=336)
    ])
    input_dict = transforms(input_dict)
    img = input_dict['img']
    mask = input_dict['mask']
    img_data = NestedTensor(img.unsqueeze(0), mask)

    #process text data
    examples = read_examples(phrase, idx)
    features = convert_examples_to_features(examples=examples, seq_length=128, tokenizer=tokenizer)
    raw_batch1 = features[0].input_ids
    raw_batch2 = features[0].input_mask
    word_id = torch.tensor(np.array(raw_batch1))
    word_mask = torch.tensor(np.array(raw_batch2))
    # phrase_data = NestedTensor(phrase, None)
    img_data = img_data.to(device)
    phrase_data = NestedTensor(word_id.unsqueeze(0), word_mask.unsqueeze(0)).to(device)
    pred_res = model(img_data,phrase_data ,{})

    print("Success")

    #code to display pred_Res on input image
    draw_img = ImageDraw.Draw(pil_img)
    bbox = pred_res[0].tolist()
    bbox_coords = [0]*4
    bbox_coords[0] = bbox[0]*pil_img.width
    bbox_coords[1] = bbox[1]*pil_img.height
    bbox_coords[2] = bbox[2]*pil_img.width//2
    bbox_coords[3] = bbox[3]*pil_img.height//2
    bbox_updated_coords = [bbox_coords[0]-bbox_coords[2], bbox_coords[1],bbox_coords[0], bbox_coords[1]+bbox_coords[3]]
    # bbox = [int(x*imsize) for x in bbox]
    # bbox_coords = [bbox[0] - bbox[2]//2, bbox[1], bbox[0], bbox[1] + bbox[3]//2]
    # bbox = [bbox[0] , bbox[1] + bbox[3], bbox[0], bbox[1]]
    draw_img.rectangle(bbox_updated_coords, outline ="red")
    pil_img.show()
    # convert_tensor = T.ToTensor()
    # img_tensor = convert_tensor(pil_img)
    # image_to_disp = draw_bounding_boxes(img_tensor, pred_res[0].unsqueeze(0), width=2, colors='red')
    # #show image_to_disp
    
    # plt.show(image_to_disp.permute(1, 2, 0))
    # image_to_disp.show()
    return None

def main(args, config):
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    print("Inside main function")
    start_epoch = 0
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']
    #### Dataset ####
    print("Creating dataset")

    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)
    #### Model ####
    print("Creating model")
    model = MPLUG(config = config, text_encoder=args.text_encoder, text_decoder=args.text_decoder, tokenizer=tokenizer)
    model = model.to(device)
    load_checkpoint(model,args.checkpoint,args,config)
    evaluate(model, device, tokenizer)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/grounding_mplug_large.yaml')
    parser.add_argument('--checkpoint', default='./mplug_base.pth')
    parser.add_argument('--eval_checkpoint', default='')
    parser.add_argument('--output_dir', default='output/RefCOCO')   
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--text_decoder', default='bert-base-uncased')
    parser.add_argument('--dataset', default='vg_uni')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--min_length', default=1, type=int)
    parser.add_argument('--max_length', default=10, type=int)
    parser.add_argument('--beam_size', default=5, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--do_two_optim', action='store_true')
    parser.add_argument('--do_amp', action='store_true')
    parser.add_argument('--no_init_decocde', action='store_true')
    parser.add_argument('--do_accum', action='store_true')
    parser.add_argument('--accum_steps', default=4, type=int)
    parser.add_argument('--finetune', action='store_true')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    config.update(vars(args))
    if args.finetune:
        config['optimizer']['lr1']=2e-6
        config['optimizer']['lr2']=2e-6
    if 'clip_name' not in config:
        config['clip_name'] = 'ViT-B-16.tar'
    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)


    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    main(args, config)
