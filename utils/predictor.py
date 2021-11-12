import configparser
from utils.dataset import *
from utils.network import *
from utils.utils_predict import *
import os, errno
import seaborn as sns
import matplotlib.pyplot as plt
import copy
import argparse
import PIL
from zipfile import ZipFile
from glob import glob
import openpifpaf.datasets as datasets


DOWNLOAD = None
INPUT_SIZE=51
FONT = cv2.FONT_HERSHEY_SIMPLEX

print('OpenPifPaf version', openpifpaf.__version__)
print('PyTorch version', torch.__version__)


class Predictor():
    """
        Class definition for the predictor.
    """
    def __init__(self, args, pifpaf_ver='shufflenetv2k30'):
        device = args.device
        args.checkpoint = pifpaf_ver
        args.force_complete_pose = True
        if device != 'cpu':
            use_cuda = torch.cuda.is_available()
            self.device = torch.device("cuda:{}".format(device) if use_cuda else "cpu")
        else:
            self.device = torch.device('cpu')
        args.device = self.device
        print('device : {}'.format(self.device))
        self.path_images = args.images
        self.net, self.processor, self.preprocess = load_pifpaf(args)
        self.path_model = './models/predictor'
        try:
            os.makedirs(self.path_model)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        self.mode = args.mode
        self.model = self.get_model()
        self.path_out = './output'
        self.path_out = filecreation(self.path_out)

    
    def get_model(self):
        if self.mode == 'joints':
            model = LookingModel(INPUT_SIZE)
            if not os.path.isfile(os.path.join(self.path_model, 'LookingModel_LOOK+PIE.p')):
                """
                DOWNLOAD(LOOKING_MODEL, os.path.join(self.path_model, 'Looking_Model.zip'), quiet=False)
                with ZipFile(os.path.join(self.path_model, 'Looking_Model.zip'), 'r') as zipObj:
                    # Extract all the contents of zip file in current directory
                    zipObj.extractall()
                exit(0)"""
                raise NotImplementedError
            model.load_state_dict(torch.load(os.path.join(self.path_model, 'LookingModel_LOOK+PIE.p')))
            model.eval()
        else:
            model = AlexNet_head(self.device)
            if not os.path.isfile(os.path.join(self.path_model, 'AlexNet_LOOK.p')):
                """
                DOWNLOAD(LOOKING_MODEL, os.path.join(self.path_model, 'Looking_Model.zip'), quiet=False)
                with ZipFile(os.path.join(self.path_model, 'Looking_Model.zip'), 'r') as zipObj:
                    # Extract all the contents of zip file in current directory
                    zipObj.extractall()
                exit(0)"""
                raise NotImplementedError
            model.load_state_dict(torch.load(os.path.join(self.path_model, 'AlexNet_LOOK.p')))
            model.eval()
        return model

    def predict_look(self, boxes, keypoints, im_size):
        label_look = []
        final_keypoints = []
        if len(boxes) != 0:
            for i in range(len(boxes)):
                kps = keypoints[i]
                kps_final = np.array([kps[0], kps[1], kps[2]]).flatten().tolist()
                X, Y = kps_final[:17], kps_final[17:34]
                X, Y = normalize_by_image_(X, Y, im_size)
                #X, Y = normalize(X, Y, divide=True, height_=False)
                kps_final_normalized = np.array([X, Y, kps_final[34:]]).flatten().tolist()
                final_keypoints.append(kps_final_normalized)
            tensor_kps = torch.Tensor([final_keypoints])#
            out_labels = self.model(tensor_kps.squeeze(0)).detach().cpu().numpy().reshape(-1)
        else:
            out_labels = []
        return out_labels
    
    def predict_look_alexnet(self, boxes, image):
        out_labels = []
        data_transform = transforms.Compose([
                        SquarePad(),
                        transforms.Resize((227,227)),
                    transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])])
        if len(boxes) != 0:
            for i in range(len(boxes)):
                bbox = boxes[i]
                x1, y1, x2, y2, _ = bbox
                w, h = abs(x2-x1), abs(y2-y1)
                head_image = Image.fromarray(np.array(image)[int(y1):int(y1+(h/3)), int(x1):int(x2), :])
                head_tensor = data_transform(head_image).unsqueeze(0).to(self.device)
                out_labels.append(self.model(head_tensor).detach().cpu().numpy().reshape(-1))
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
            else:
                color = (0,0,255)
            mask = draw_skeleton(mask, keypoints[i], color)
        mask = cv2.erode(mask,(7,7),iterations = 1)
        mask = cv2.GaussianBlur(mask,(3,3),0)
        #open_cv_image = cv2.addWeighted(open_cv_image, 0.5, np.ones(open_cv_image.shape, dtype=np.uint8)*255, 0.5, 1.0)
        #open_cv_image = cv2.addWeighted(open_cv_image, 0.5, np.zeros(open_cv_image.shape, dtype=np.uint8), 0.5, 1.0)
        open_cv_image = cv2.addWeighted(open_cv_image, 1, mask, transparency, 1.0)
        cv2.imwrite(os.path.join(self.path_out, image_name[:-4]+'.pedictions.png'), open_cv_image)

    def predict(self, args):
        transparency = args.transparency
        eyecontact_thresh = args.looking_threshold
        
        if args.glob:
            array_im = glob(os.path.join(args.images[0], '*'+args.glob))
        else:
            array_im = args.images
        
        data = datasets.ImageList(array_im, preprocess=self.preprocess)
        loader = torch.utils.data.DataLoader(data, batch_size=1, pin_memory=False, collate_fn=openpifpaf.datasets.collate_images_anns_meta)
        
        for images_batch, _, meta_batch in loader:
            pred_batch = self.processor.batch(self.net, images_batch, device=self.device)
            for idx, (pred, meta) in enumerate(zip(pred_batch, meta_batch)):
                pred = [ann.inverse_transform(meta) for ann in pred]
                with open(meta_batch[0]['file_name'], 'rb') as f:
                    cpu_image = PIL.Image.open(f).convert('RGB')
                    pifpaf_outs = {
                        'pred': pred,
                        'left': [ann.json_data() for ann in pred],
                        'image': cpu_image}
                break
            im_name = os.path.basename(meta['file_name'])
            im_size = (cpu_image.size[0], cpu_image.size[1])
            boxes, keypoints = preprocess_pifpaf(pifpaf_outs['left'], im_size, enlarge_boxes=False)
            if self.mode == 'joints':
                pred_labels = self.predict_look(boxes, keypoints, im_size)
            else:
                pred_labels = self.predict_look_alexnet(boxes, cpu_image)
            self.render_image(pifpaf_outs['image'], boxes, keypoints, pred_labels, im_name, transparency, eyecontact_thresh)

    
