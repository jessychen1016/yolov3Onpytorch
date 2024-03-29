#!/home/jessy104/miniconda3/envs/yolo3pytorch/bin/python3
import argparse
from sys import platform
import rospy
import cv_bridge
from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *
import sys
from yolov3_ros.msg import image_with_class

def detect(save_txt=False, save_img=False):
    img_size = (320, 192) if ONNX_EXPORT else opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
    out, source, rostopic_color, rostopic_depth, weights, half, view_img = opt.output, opt.source, opt.rostopic_color, opt.rostopic_depth, opt.weights, opt.half, opt.view_img
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')
    rosFlag = source == '1' or source.startswith('/')

    # Initialize
    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder

    # Initialize model
    model = Darknet(opt.cfg, img_size)

    # Load weights
    attempt_download(weights)
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        _ = load_darknet_weights(model, weights)

    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Fuse Conv2d + BatchNorm2d layers
    # model.fuse()

    # Eval mode
    model.to(device).eval()

    # Export mode
    if ONNX_EXPORT:
        img = torch.zeros((1, 3) + img_size)  # (1, 3, 320, 192)
        torch.onnx.export(model, img, 'weights/export.onnx', verbose=False, opset_version=11)

        # Validate exported model
        import onnx
        model = onnx.load('weights/export.onnx')  # Load the ONNX model
        onnx.checker.check_model(model)  # Check that the IR is well formed
        print(onnx.helper.printable_graph(model.graph))  # Print a human readable representation of the graph
        return

    # Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(sources=source, img_size=img_size, half=half)
    elif rosFlag:
        view_img = True
        torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadRosTopic(path=source, img_size=img_size, half=half, rostopic_color=rostopic_color, rostopic_depth=rostopic_depth)
    else:
        save_img = True
        dataset = LoadImages(path=source, img_size=img_size, half=half)
    # Get classes and colors
    classes = load_classes(parse_data_cfg(opt.data)['names'])
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]
    # Run inference
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        t = time.time()

        # Get detections
        img = torch.from_numpy(img).to(device)
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred = model(img)[0]

        if opt.half:
            pred = pred.float()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.nms_thres)

        # Apply
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)


        # an image_with_class variable
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i]
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            s += '%gx%g ' % img.shape[2:]  # print string
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # get color and depth topic
                color_msg = dataset.get_color()
                depth_msg = dataset.get_depth()

                #to create a variable to publish
                class_location = image_with_class()
                class_location.ColorImage = color_msg
                class_location.DepthImage = depth_msg
                
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, classes[int(c)])  # add to string

                # Write results
                for *xyxy, conf, _, cls in det:
                    if save_txt:  # Write to file
                        with open(save_path + '.txt', 'a') as file:
                            file.write(('%g ' * 6 + '\n') % (*xyxy, cls, conf))

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (classes[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])

                    # label.split(None,1)[0] is the class name
                    # label.split(None,1)[1] is the trusting confident
                    # int(xyxy[0]) is the x_position_of_the_box_DownLeft_corner
                    # int(xyxy[1]) is the y_position_of_the_box_DownLeft_corner
                    # int(xyxy[2]) is the x_position_of_the_box_UpRight_corner
                    # int(xyxy[3]) is the y_position_of_the_box_UpRight_corner
                    class_location.class_name_of_the_box.append(label.split(None,1)[0])
                    class_location.x_position_of_the_box_DownLeft_corner.append(str(int(xyxy[0]))+" ")
                    class_location.y_position_of_the_box_DownLeft_corner.append(str(int(xyxy[1]))+" ")
                    class_location.x_position_of_the_box_UpRight_corner.append(str(int(xyxy[2]))+" ")
                    class_location.y_position_of_the_box_UpRight_corner.append(str(int(xyxy[3]))+" ")
            print('%sDone. (%.3fs)' % (s, time.time() - t))

            # publish class_and_image
            if det is not None and len(det):
                class_image.publish(class_location)
            
            # Stream results
            if view_img:
                cv2.imshow(p, im0)

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
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + out + ' ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))




if __name__ == '__main__':
    rospy.init_node('Detecter', anonymous=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='cfg file path')
    parser.add_argument('--data', type=str, default='data/coco.data', help='coco.data file path')
    parser.add_argument('--weights', type=str, default='weights/colorDown/1209best.pt', help='path to weights file')
    parser.add_argument('--source', type=str, default='1', help='source')  # input file/folder, 0 for webcam, 1 for ros
    parser.add_argument('--rostopic_color', type=str, default='/AeroCameraDown/infra2/image_rect_raw', help='source') # set subscribe topic if the source value =1
    parser.add_argument('--rostopic_depth', type=str, default='/AeroCameraDown/depth/image_rect_raw', help='source') # set subscribe topic if the source value =1
    parser.add_argument('--output', type=str, default='output/try', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=1280, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.6, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    opt = parser.parse_args()
    print(opt)

    # global color_pub
    # global depth_pub
    global class_image
    # color_pub = rospy.Publisher("colorYOLO", Image, queue_size=1)
    # depth_pub = rospy.Publisher("depthYOLO", Image, queue_size=1)
    class_image = rospy.Publisher("class_image_YOLO", image_with_class, queue_size=1)

    with torch.no_grad():
        detect()
    rospy.spin()