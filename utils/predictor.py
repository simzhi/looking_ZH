import os, errno
import matplotlib.pyplot as plt
import argparse
import PIL
from glob import glob
from tqdm import tqdm
import time

from utils.network import *
from utils.utils_predict import *

from PIL import Image, ImageFile

DOWNLOAD = None
INPUT_SIZE = 51
FONT = cv2.FONT_HERSHEY_SIMPLEX

ImageFile.LOAD_TRUNCATED_IMAGES = True

print('OpenPifPaf version', openpifpaf.__version__)
print('PyTorch version', torch.__version__)


class Predictor():
    """
        Class definition for the predictor.
    """
    def __init__(self, args, pifpaf_ver='shufflenetv2k30'):
        device = args.device
        source = args.source
        source = str(source)
        webcam = source.isnumeric()
        args.checkpoint = pifpaf_ver
        args.force_complete_pose = True
        # cuda Device or CPU
        if device != 'cpu':
            use_cuda = torch.cuda.is_available()
            self.device = torch.device("cuda:{}".format(device) if use_cuda else "cpu")
        else:
            self.device = torch.device('cpu')
        args.device = self.device
        print('device : {}'.format(self.device))
        # images path
        self.path_images = args.images
        print('Images path : {}'.format(self.path_images))
        # self.net, self.processor, self.preprocess = load_pifpaf(args)
        self.predictor_ = load_pifpaf(args)
        self.path_model = './models/predictor'
        try:
            os.makedirs(self.path_model)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        self.mode = args.mode
        self.model = self.get_model().to(self.device)
        if args.image_output is None:
            self.path_out = './output'
            self.path_out = filecreation(self.path_out)
        else:
            self.path_out = args.image_output
        self.track_time = args.time
        if self.track_time:
            self.pifpaf_time = []
            self.inference_time = []
            self.total_time = []

    
    def get_model(self):
        if self.mode == 'joints':
            model = LookingModel(INPUT_SIZE)
            print(self.device)
            if not os.path.isfile(os.path.join(self.path_model, 'LookingModel_LOOK+PIE.p')):
                raise NotImplementedError
            model.load_state_dict(torch.load(os.path.join(self.path_model, 'LookingModel_LOOK+PIE.p'), map_location=self.device))
            model.eval()
        return model

    def predict_look(self, boxes, keypoints, im_size, batch_wise=True):
        label_look = []
        final_keypoints = []
        if batch_wise:
            if len(boxes) != 0:
                for i in range(len(boxes)):
                    kps = keypoints[i]
                    kps_final = np.array([kps[0], kps[1], kps[2]]).flatten().tolist()
                    X, Y = kps_final[:17], kps_final[17:34]
                    X, Y = normalize_by_image_(X, Y, im_size)
                    #X, Y = normalize(X, Y, divide=True, height_=False)
                    kps_final_normalized = np.array([X, Y, kps_final[34:]]).flatten().tolist()
                    final_keypoints.append(kps_final_normalized)
                tensor_kps = torch.Tensor([final_keypoints]).to(self.device)
                if self.track_time:
                    start = time.time()
                    out_labels = self.model(tensor_kps.squeeze(0)).detach().cpu().numpy().reshape(-1)
                    end = time.time()
                    self.inference_time.append(end-start)
                else:
                    out_labels = self.model(tensor_kps.squeeze(0)).detach().cpu().numpy().reshape(-1)
            else:
                out_labels = []
        else:
            if len(boxes) != 0:
                for i in range(len(boxes)):
                    kps = keypoints[i]
                    kps_final = np.array([kps[0], kps[1], kps[2]]).flatten().tolist()
                    X, Y = kps_final[:17], kps_final[17:34]
                    X, Y = normalize_by_image_(X, Y, im_size)
                    #X, Y = normalize(X, Y, divide=True, height_=False)
                    kps_final_normalized = np.array([X, Y, kps_final[34:]]).flatten().tolist()
                    #final_keypoints.append(kps_final_normalized)
                    tensor_kps = torch.Tensor(kps_final_normalized).to(self.device)
                    if self.track_time:
                        start = time.time()
                        out_labels = self.model(tensor_kps.unsqueeze(0)).detach().cpu().numpy().reshape(-1)
                        end = time.time()
                        self.inference_time.append(end-start)
                    else:
                        out_labels = self.model(tensor_kps.unsqueeze(0)).detach().cpu().numpy().reshape(-1)
            else:
                out_labels = []
        return out_labels
    
    def render_image(self, image, bbox, keypoints, pred_labels, image_name, transparency, eyecontact_thresh):
        open_cv_image = np.array(image) 
        open_cv_image = open_cv_image[:, :, ::-1].copy()
        
        scale = 0.007
        imageWidth, imageHeight, _ = open_cv_image.shape
        font_scale = min(imageWidth,imageHeight)/(10/scale)


        mask = np.zeros(open_cv_image.shape, dtype=np.uint8)
        for i, label in enumerate(pred_labels):

            if label > eyecontact_thresh:
                color = (0,255,0)
                print('Eye Contact Prediction : True {}'.format(label))
            else:
                color = (0,0,255)
                print('Eye Contact Prediction : False {}'.format(label))
            mask = draw_skeleton(mask, keypoints[i], color)
        mask = cv2.erode(mask,(7,7),iterations = 1)
        mask = cv2.GaussianBlur(mask,(3,3),0)
        #open_cv_image = cv2.addWeighted(open_cv_image, 0.5, np.ones(open_cv_image.shape, dtype=np.uint8)*255, 0.5, 1.0)
        #open_cv_image = cv2.addWeighted(open_cv_image, 0.5, np.zeros(open_cv_image.shape, dtype=np.uint8), 0.5, 1.0)
        open_cv_image = cv2.addWeighted(open_cv_image, 1, mask, transparency, 1.0)
        cv2.imwrite(os.path.join(self.path_out, image_name[:-4]+'.predictions.png'), open_cv_image)


    def predict(self, args):
        transparency = args.transparency
        eyecontact_thresh = args.looking_threshold
        
        if args.glob:
            array_im = glob(os.path.join(args.images[0], '*'+args.glob))
        else:
            array_im = args.images
        loader = self.predictor_.images(array_im)
        start_pifpaf = time.time()
        for i, (pred_batch, _, meta_batch) in enumerate(tqdm(loader)):
            if self.track_time:
                end_pifpaf = time.time()
                self.pifpaf_time.append(end_pifpaf-start_pifpaf)
            cpu_image = PIL.Image.open(open(meta_batch['file_name'], 'rb')).convert('RGB')
            pifpaf_outs = {
            'json_data': [ann.json_data() for ann in pred_batch],
            'image': cpu_image
            }
            
            im_name = os.path.basename(meta_batch['file_name'])
            im_size = (cpu_image.size[0], cpu_image.size[1])
            boxes, keypoints = preprocess_pifpaf(pifpaf_outs['json_data'], im_size, enlarge_boxes=False)
            if self.mode == 'joints':
                pred_labels = self.predict_look(boxes, keypoints, im_size)
            else:
                pred_labels = self.predict_look_alexnet(boxes, cpu_image)
            if self.track_time:
                end_process = time.time()
                self.total_time.append(end_process - start_pifpaf)
            
            #break
            if self.track_time:
                start_pifpaf = time.time()
            else:
                self.render_image(pifpaf_outs['image'], boxes, keypoints, pred_labels, im_name, transparency, eyecontact_thresh)

            #if i > 20:
            #    break
        
        if self.track_time and len(self.pifpaf_time) != 0 and len(self.inference_time) != 0:
            print('Av. pifpaf time : {} ms. ± {} ms'.format(np.mean(self.pifpaf_time)*1000, np.std(self.pifpaf_time)*1000))
            print('Av. inference time : {} ms. ± {} ms'.format(np.mean(self.inference_time)*1000, np.std(self.inference_time)*1000))
            print('Av. total time : {} ms. ± {} ms'.format(np.mean(self.total_time)*1000, np.std(self.total_time)*1000))
    
