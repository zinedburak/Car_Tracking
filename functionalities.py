import torch.backends.cudnn as cudnn
from torch.cuda import is_available
from PyQt5 import QtCore
from PyQt5.QtGui import QImage, QPixmap

from utils import google_utils
from utils.datasets import *
from utils.utils import *
import os
from PyQt5 import QtGui, QtMultimedia

print(is_available())

""" TO DO :

# Train your own YOLO V5 weight for only cars bikes trucks and buses
# Create an original object tracking algorithm
# Find an object and give it an id
# Compare all the objects in two or three frames. 
# If the bounding boxes of the objects are similar give the object in the current frame the same id as the previews one

"""


def file_exists(file_path):
    file_path = file_path.replace("\\", "/")
    return os.path.isfile(file_path)


def play(GUI):
    GUI.player.play()


def pause(GUI):
    GUI.player.pause()


def openFile(GUI):
    GUI.file_name = GUI.ui.lineEdit_originalFilePath.text()
    GUI.file_name = GUI.file_name.replace("\\", "/")
    GUI.player.setMedia(QtMultimedia.QMediaContent(QtCore.QUrl.fromLocalFile(GUI.file_name)))
    GUI.player.setVideoOutput(GUI.ui.widget_Video)
    GUI.player.play()
    GUI.ui.button_play.setEnabled(True)
    GUI.button_pause.setEnabled(True)


def get_ThreshHold(GUI):
    conf_th = GUI.spinBox_conf_th.value() * 0.01
    if conf_th > 0.2:
        return conf_th
    else:
        return 0.5


def clear_Text(GUI):
    GUI.textfield_outputText.clear()


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def detect(GUI, classes=[0,2,3,5,7], augment=False, agnostic_nms=False, save_txt=True, view_img=False, device='',
           fourcc='mp4v', iou_thres=0.5, img_size=640
           , output='inference/output', source='0', weights='weights/yolov5s.pt', save_img=False):
    out, source, weights, view_img, save_txt, imgsz = \
        output, source, weights, view_img, save_txt, img_size
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')
    """Change"""
    sum = 0
    total_count = 0
    total_qt_cost = 0
    total_vis_cost = 0

    GUI.textfield_outputText.setEnabled(True)
    GUI.textfield_outputText.insertPlainText("Loading models and video \n")

    conf_thres = get_ThreshHold(GUI)
    if file_exists(GUI.lineEdit_originalFilePath.text()):
        source = GUI.lineEdit_originalFilePath.text()
        webcam = 0
        # Initialize
        device = torch_utils.select_device(device)
        if os.path.exists(out):
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder
        half = device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        google_utils.attempt_download(weights)
        model = torch.load(weights, map_location=device)['model'].float()  # load to FP32
        # torch.save(torch.load(weights, map_location=device), weights)  # update model if SourceChangeWarning
        # model.fuse()
        model.to(device).eval()
        imgsz = check_img_size(imgsz, s=model.model[-1].stride.max())  # check img_size
        if half:
            model.half()  # to FP16

        # Second-stage classifier
        classify = False
        if classify:
            modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
            modelc.to(device).eval()

        # Set Dataloader
        vid_path, vid_writer = None, None
        if webcam:
            view_img = True
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz)
        else:
            save_img = True
            view_img = True
            dataset = LoadImages(source, img_size=imgsz)

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

        # Run inference
        t0 = time.time()
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = torch_utils.time_synchronized()
            pred = model(img, augment=augment)[0]
            # print(type((pred)))

            # Apply NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes,
                                       agnostic=agnostic_nms)
            t2 = torch_utils.time_synchronized()

            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            # Process detections
            tvcstart = time.time()
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
                else:
                    p, s, im0 = path, '', im0s

                save_path = str(Path(out) / Path(p).name)
                txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    """
                    try:
                        print(torch.unique(det[:, -1]))
                    except Exception as e:
                        print(e)
                    """
                    """
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, names[int(c)])  # add to string
                    """
                    # Write results
                    for *xyxy, conf, cls in det:
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                        if save_img or view_img:  # Add bbox to image
                            label = '%s %.2f' % (names[int(cls)], conf)
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                # Print time (inference + NMS)
                GUI.textfield_outputText.insertPlainText('%sDone. (%.3fs) \n' % (s, t2 - t1))
                """Change"""
                sum += t2 - t1
                total_count += 1
                GUI.textfield_outputText.moveCursor(QtGui.QTextCursor.End)

                # Stream results
                if view_img:
                    # cv2.imshow(p, im0)
                    tbshow = time.time()
                    im0 = image_resize(im0, width=980)
                    img = QImage(im0, im0.shape[1], im0.shape[0], QImage.Format_BGR888)
                    pixMap = QPixmap(img)
                    GUI.image_frame.setPixmap(pixMap)
                    teshow = time.time()
                    total_qt_cost += (teshow - tbshow)
                    if cv2.waitKey(1) == ord('q'):  # q to quit
                        return 0

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'images':
                        cv2.imwrite(save_path, im0)
                    else:
                        if vid_path != save_path:  # new video
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # release previous video writer

                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                        vid_writer.write(im0)
            tvcend = time.time()
            total_vis_cost += (tvcend - tvcstart)

        if save_txt or save_img:
            # print('Results saved to %s' % os.getcwd() + os.sep + out)
            if platform == 'darwin':  # MacOS
                os.system('open ' + save_path)
        GUI.textfield_outputText.insertPlainText('Done. (%.3fs) \n' % (time.time() - t0))
        GUI.textfield_outputText.insertPlainText('Total Frames in the video:  (%.3fs) \n' % total_count)
        GUI.textfield_outputText.insertPlainText('average frame is processed in  (%.3fs) \n' % (sum / total_count))
        GUI.textfield_outputText.insertPlainText(
            'average Pyqt cost per frame is (%.3fs) \n' % (total_qt_cost / total_count))
        GUI.textfield_outputText.insertPlainText('total qt cost (%.3fs) \n' % total_qt_cost)
        GUI.textfield_outputText.insertPlainText(
            'average visualization  cost per frame is (%.3fs) \n' % (total_vis_cost / total_count))
        GUI.textfield_outputText.insertPlainText('total visualization cost (%.3fs) \n' % total_vis_cost)
        # print('Done. (%.3fs)' % (time.time() - t0))
        GUI.textfield_outputText.insertPlainText("------------------------------- \n")

    else:
        GUI.textfield_outputText.insertPlainText("File Path is not valid please insert a valid path \n")