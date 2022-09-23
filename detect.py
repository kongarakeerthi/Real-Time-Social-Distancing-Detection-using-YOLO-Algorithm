import argparse

from utils.datasets import *
from utils.utils import *


def find(weights='weights/yolov5s.pt', source='storage/images',
           output='storage/output', img_size=640, conf_thres=0.4, iou_thres=0.5, fourcc='mp4v',
           device=''):
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    # parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    # parser.add_argument('--augment', action='store_true', help='augmented inference')

    # parser_val = parser.parse_args()
    device = torch_utils.check_device(device)
    if os.path.exists(output):
        shutil.rmtree(output)  
    os.makedirs(output)  
    half_device = False
    img_size = convert_img(img_size)

    google_utils.attempt_download(weights)
    model = torch.load(weights, map_location=device)[
        'model'].float()  
    model.to(device).eval()
    if half_device:
        model.half()  

    dataset = convert_data_img(source, img_size=img_size)

    names = model.names if hasattr(model, 'names') else model.modules.names

    v_p, v_w = None, None
    input_image = torch.zeros((1, 3, img_size, img_size), device=device) 
    for path, input_image, img, video_capture in dataset:
        input_image = torch.from_numpy(input_image).to(device)
        input_image = input_image.half() if half_device else input_image.float()  
        input_image /= 255.0 
        if input_image.ndimension() == 3:
            input_image = input_image.unsqueeze(0)

        predictions = model(input_image)[0]
        predictions = nms(predictions, conf_thres, iou_thres,fast=True)

        coordinates_sapiens = []

        for  detections in predictions:  
            
            to_save_path = str(Path(output) / Path(path).name)
            if detections is not None and len(detections):
                detections[:, :4] = rescalecordinates(
                    input_image.shape[2:], detections[:, :4], img.shape).round() 
                for *cordinate, confidence, classes in detections:
                        label = '%s %.2f' % (names[int(classes)], confidence)
                        if label is not None:
                            if (label.split())[0] == 'person':
                                coordinates_sapiens.append(cordinate)
                                # plot_one_box(xyxy, im0, line_thickness=3)
                                sapiens_round_plot(cordinate, img)

    
            plot_distance(coordinates_sapiens, img, dist_thres_lim=(200, 250))

            if v_p != to_save_path:  # new video
                v_p = to_save_path
                if isinstance(v_w, cv2.VideoWriter):
                    v_w.release()  # release previous video writer

                fps = video_capture.get(cv2.CAP_PROP_FPS)
                w = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                v_w = cv2.VideoWriter(
                    to_save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
            v_w.write(img)