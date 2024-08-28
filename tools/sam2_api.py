import torch
import numpy as np
import cv2
from sam2.build_sam import build_sam2_video_predictor
import shutil
import base64
import io

import os
from flask import Flask, jsonify, request

def encode_ndarray(data):
    buffer = io.BytesIO()
    np.save(buffer, data)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')

def decode_ndarray(data):
    decoded_image = base64.b64decode(data)
    buffer = io.BytesIO(decoded_image)
    return np.load(buffer)

app = Flask(__name__)

video_folder = "./tmp_data"
sam2_checkpoint = "./checkpoints/sam2_hiera_tiny.pt"
sam2_model_cfg = "./sam2_hiera_t.yaml"
image_counter = 0
memory_length = 7
sam2_video_predictor = build_sam2_video_predictor(sam2_model_cfg, sam2_checkpoint)

init = False
sam2_inference_state = None

#TO UPDATE

def prepare_image_for_sam(
        image,
        image_size,
        img_mean=(0.485, 0.456, 0.406),
        img_std=(0.229, 0.224, 0.225),
    ):
        img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
        img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]
        video_height, video_width, _ = image.shape
        image = cv2.resize(image, (image_size, image_size))
        if image.dtype == np.uint8:  # np.uint8 is expected for JPEG images
            image = image / 255.0
        else:
            raise RuntimeError(f"Unknown image dtype: {image.dtype}")
        image = torch.from_numpy(image).permute(2, 0, 1)
        image -= img_mean
        image /= img_std
        return image, video_height, video_width

@app.route('/init_segmentation', methods=['POST'])
def initialize_objects_identification():
    '''
        interactively set up all objects with SAM
    '''
    global image_counter, sam2_inference_state
    prompts = request.json.get('prompts')
    image = decode_ndarray(request.json.get('image'))
    results = {'obj_ids':[], 'masks':[]}
    
    #save initial image for SAM
    if os.path.exists(video_folder):
        shutil.rmtree(video_folder)
    os.mkdir(video_folder)
    
    cv2.imwrite(os.path.join(video_folder, '0.jpg'), image)
    image_counter = 1
    #set up SAM model and store segmentation in the objects
    sam2_inference_state = sam2_video_predictor.init_state(video_path=video_folder)

    for ann_obj_id, prompt in enumerate(prompts):
        # Let's add a 2nd positive click at (x, y) = (250, 220) to refine the mask
        # sending all clicks (and their labels) to `add_new_points`
        # points = np.array([[210, 350], [250, 220]], dtype=np.float32)
        # for labels, `1` means positive click and `0` means negative click
        points = np.array(prompt['positive_points'] + prompt['negative_points'])
        labels = np.array([1]*len(prompt['positive_points']) + [0]*len(prompt['negative_points']), np.int32)
        _, out_obj_ids, out_mask_logits = sam2_video_predictor.add_new_points(
            inference_state=sam2_inference_state,
            frame_idx=0,
            obj_id=ann_obj_id,
            points=points,
            labels=labels,
        )
        
    for i, out_obj_id in enumerate(out_obj_ids):
        results['obj_ids'].append(out_obj_id)
        results['masks'].append(encode_ndarray(out_mask_logits[i].cpu().numpy()))
    
    return jsonify(results)

@app.route('/update_segmentation', methods=['POST'])
def update_segmentation():
    '''
        estimate segmentation for the objects
    '''
    global image_counter, sam2_inference_state
    image = decode_ndarray(request.json.get('image'))
    results = {'obj_ids':[], 'masks':[]}

    #update inference_state
    # image_path = os.path.join(video_folder, str(image_counter)+'.jpg')
    # cv2.imwrite(image_path, image)
    # image_counter += 1
    image, _, _ = prepare_image_for_sam(image, sam2_video_predictor.image_size)
    image = image.to(sam2_inference_state['images'].device)

    if sam2_inference_state['num_frames'] < memory_length:
        sam2_inference_state['images'] = torch.cat([sam2_inference_state['images'], image[None,:]], dim=0)
        sam2_inference_state['num_frames'] += 1
    else:
        sam2_inference_state['images'] = torch.cat([sam2_inference_state['images'][0][None,:], sam2_inference_state['images'][-memory_length+2:], image[None,:]], dim=0)
        sam2_inference_state['num_frames'] = memory_length
        for i in range(1,memory_length-1):
            sam2_inference_state['output_dict']['non_cond_frame_outputs'][i] = sam2_inference_state['output_dict']['non_cond_frame_outputs'][i+1]
        sam2_inference_state['cached_features'] = {}
    #Not sliding window currently, need to implement a sliding window version
        sam2_inference_state['images'] = torch.cat([sam2_inference_state['images'], image[None,:]], dim=0)
    #SAM model propograte
    sam_results = sam2_video_predictor.propagate_in_video(sam2_inference_state, start_frame_idx=sam2_inference_state['num_frames']-1)
    
    for out_frame_idx, out_obj_ids, out_mask_logits in sam_results:
        for i, out_obj_id in enumerate(out_obj_ids):
            results['obj_ids'].append(out_obj_id)
            results['masks'].append(encode_ndarray(out_mask_logits[i].cpu().numpy()))
    
    return jsonify(results)
      
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)