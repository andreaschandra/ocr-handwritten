import time
import yaml
from argparse import Namespace
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import matplotlib.pyplot as plt
import cv2
from skimage import io
import numpy as np
import craft_utils
import imgproc
from craft import CRAFT
from collections import OrderedDict

def load_config():
    with open('configs/craft.yaml', 'r') as f:
        args = yaml.safe_load(f)

    return args

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, args, refine_net=None):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args['canvas_size'], interpolation=cv2.INTER_LINEAR, mag_ratio=args['mag_ratio'])
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = x.unsqueeze(0)                          # [c, h, w] to [b, c, h, w]

    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)
    
    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    return boxes, polys, ret_score_text

def load_model_detection(args):

    args['cuda'] = True if torch.cuda.is_available() else False

    # load net
    net = CRAFT()     # initialize

    print('Loading weights from checkpoint (' + args['trained_model'] + ')')
    if args['cuda']:
        net.load_state_dict(copyStateDict(torch.load(args['trained_model'])))
    else:
        net.load_state_dict(copyStateDict(torch.load(args['trained_model'], map_location='cpu')))

    if args['cuda']:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()

    return net

def predict_bboxes(image, args, net, refine_net=None):

    image = imgproc.convertImage(image)

    bboxes, polys, score_text = test_net(net, image, args['text_threshold'], args['link_threshold'], args['low_text'], args['cuda'], args['poly'], args, refine_net)

    return bboxes, polys, score_text

def extract_bboxes(bboxes):

    rect_bboxes = []
    for point in bboxes:
        point = point.astype(np.int32)
        x1, x2 = point[:, 0].min(), point[:, 0].max()
        y1, y2 = point[:, 1].min(), point[:, 1].max()
        rect_bboxes.append([y1, y2, x1, x2])

    rect_bboxes = np.array(rect_bboxes)

    return rect_bboxes

def get_lines(rect_bboxes):
    
    threshold = 20
    lines, entity = [], []
    entity.append(rect_bboxes[0])
    for idx in range(len(rect_bboxes)-1):
        distance = abs(rect_bboxes[idx][0] - rect_bboxes[idx+1][0])
        if distance < threshold:
            entity.append(rect_bboxes[idx+1])
        else:
            lines.append(entity)
            entity = []
            entity.append(rect_bboxes[idx+1])
            
    return lines

def load_model_recognition():

    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
    model = VisionEncoderDecoderModel.from_pretrained('scripts/model/trocr-finetuned-augmented', local_files_only=True)

    return processor, model

def recognize_text(lines, image, processor, model):

    text_lines = []
    for line in tqdm(lines):
        if 20 <= len(line) <= 30:
            text_row = []
            for bbox in line:
                sliced = image[bbox[0]: bbox[1], bbox[2]:bbox[3], :]
                sliced_im = Image.fromarray(sliced)
                pixel_values = processor(sliced_im, return_tensors="pt").pixel_values
                generated_ids = model.generate(pixel_values)
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                text_row.append(generated_text)
        
            text_lines.append(", ".join(text_row))
    text = "\n".join(text_lines)

    return text

def predict(image):
    args = load_config()
    net = load_model_detection(args)
    processor, text_model = load_model_recognition()
    bboxes, _, _ = predict_bboxes(image, args, net)
    rect_bboxes = extract_bboxes(bboxes)
    lines = get_lines(rect_bboxes)
    text = recognize_text(lines, image, processor, text_model)
    return text

if __name__ == "__main__":
    im_arr = cv2.imread('../data/data-ocr/data-ocr/0EAC26CF-CAA4-4B5B-B521-EA0E42EF650A.JPG')
    im_arr = im_arr[:, :, ::-1]
    text = predict(im_arr)
    print(text)