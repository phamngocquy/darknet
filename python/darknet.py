import os
import random
from ctypes import *
import cv2
import numpy as np
import glob
import dlib

from python.entity.Helper import Helper


def sample(probs):
    s = sum(probs)
    probs = [a / s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs) - 1


def c_array(ctype, values):
    arr = (ctype * len(values))()
    arr[:] = values
    return arr


class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]


class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]


class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]


# lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
lib = CDLL("/home/haku/Yolo/darknet/libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

# draw_detections = lib.draw_detections
# draw_detections.argtypes = [image, ]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)


def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res


def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    im = load_image(image, 0, 0)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if nms:
        do_nms_obj(dets, num, meta.classes, nms)

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_detections(dets, num)
    return res


def array_to_image(arr):
    # need to return old values to avoid python freeing memory
    arr = arr.transpose(2, 0, 1)
    c, h, w = arr.shape[0:3]
    arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
    data = arr.ctypes.data_as(POINTER(c_float))
    im = IMAGE(w, h, c, data)
    return im, arr


def detect_vid(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    """if isinstance(image, bytes):
        # image is a filename
        # i.e. image = b'/darknet/data/dog.jpg'
        im = load_image(image, 0, 0)
    else:
        # image is an nparray
        # i.e. image = cv2.imread('/darknet/data/dog.jpg')
        im, image = array_to_image(image)
        rgbgr_image(im)
    """
    im, image = array_to_image(image)
    rgbgr_image(im)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh,
                             hier_thresh, None, 0, pnum)
    num = pnum[0]
    if nms: do_nms_obj(dets, num, meta.classes, nms)

    res = []
    for j in range(num):
        a = dets[j].prob[0:meta.classes]
        if any(a):
            ai = np.array(a).nonzero()[0]
            for i in ai:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i],
                            (b.x, b.y, b.w, b.h)))

    res = sorted(res, key=lambda x: -x[1])
    if isinstance(image, bytes): free_image(im)
    free_detections(dets, num)
    return res


def detect_img(gray):
    # net = load_net("cfg/densenet201.cfg", "/home/pjreddie/trained/densenet201.weights", 0)
    # im = load_image("data/wolf.jpg", 0, 0)
    # meta = load_meta("cfg/imagenet1k.data")
    # r = classify(net, meta, im)
    # print r[:10]
    net = load_net(b"/home/haku/Yolo/darknet/cfg/yolov3.cfg", b"/home/haku/Yolo/darknet/yolov3.weights", 0)
    meta = load_meta(b"/home/haku/Yolo/darknet/cfg/coco.data")
    r = detect_vid(net, meta, gray)
    # print(r)


def detect_from_video():
    net = load_net(b"/home/haku/Yolo/darknet/cfg/yolov3.cfg", b"/home/haku/Yolo/darknet/yolov3.weights", 0)
    meta = load_meta(b"/home/haku/Yolo/darknet/cfg/coco.data")
    cap = cv2.VideoCapture('/home/haku/Yolo/darknet/data/test3.mp4')
    # cap = cv2.VideoCapture('rtsp://admin:admin@172.19.5.74:554/cam/realmonitor?channel=1&subtype=0&unicast=true
    # &proto=Onvif')
    font = cv2.FONT_HERSHEY_SIMPLEX
    my_person_list = None
    tmp_person_list = None
    print("starting")
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        r = detect_vid(net, meta, frame)
        cv2.imwrite('/home/haku/Yolo/darknet/tmp/frame%d.jpg' % count, frame)
        if my_person_list is None:
            my_person_list = Helper.get_list_person(r)
            tmp_person_list = Helper.get_list_person(r)
            for i in range(len(my_person_list)):
                my_person_list[i].identifi_code = i + 1
                tmp_person_list[i].identifi_code = i + 1
        else:
            my_person_list = Helper.get_list_person(r)
            Helper.person_mapping(my_person_list, tmp_person_list)

        for person in my_person_list:
            bottom_left_x, bottom_left_y, rcg_width, rcg_height = person.create_rectangle()
            print(bottom_left_x, bottom_left_y, rcg_width, rcg_height)
            cv2.rectangle(frame, (int(bottom_left_x), int(bottom_left_y)), (int(rcg_width), int(rcg_height)),
                          (255, 0, 0), 2)
            cv2.putText(frame, str(person.identifi_code), (int(person.center_x), int(person.center_y)), font, 1,
                        (255, 255, 255), 2,
                        cv2.LINE_AA)

        cv2.imshow('frame', frame)
        tmp_person_list = my_person_list
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        count += 1
    # print("success")
    cv2.destroyAllWindows()
    cap.release()


def tracker_obj():
    net = load_net(b"/home/haku/Yolo/darknet/cfg/yolov3.cfg", b"/home/haku/Yolo/darknet/yolov3.weights", 0)
    meta = load_meta(b"/home/haku/Yolo/darknet/cfg/coco.data")
    cap = cv2.VideoCapture('/home/haku/Yolo/darknet/data/test3.mp4')
    tracker = dlib.correlation_tracker()
    win = dlib.image_window()
    print("starting")
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        r = detect_vid(net, meta, frame)
        my_person_list = Helper.get_list_person(r)
        if count == 0:
            bottom_left_x, bottom_left_y, rcg_width, rcg_height = my_person_list[1].create_rectangle()
            print(bottom_left_x, bottom_left_y, rcg_width, rcg_height)
            tracker.start_track(frame, dlib.rectangle(int(bottom_left_x), int(bottom_left_y), int(rcg_width),
                                                      int(rcg_height)))
        else:
            tracker.update(frame)
            count += 1

        win.clear_overlay()
        win.set_image(frame)
        win.add_overlay(tracker.get_position())
    # print("success")
    cv2.destroyAllWindows()
    cap.release()


def tracker_obj_2():
    net = load_net(b"/home/haku/Yolo/darknet/cfg/yolov3.cfg", b"/home/haku/Yolo/darknet/yolov3.weights", 0)
    meta = load_meta(b"/home/haku/Yolo/darknet/cfg/coco.data")
    cap = cv2.VideoCapture('/home/haku/Yolo/darknet/data/test4.mp4')

    cv2.namedWindow("tracking")

    print('Select 3 tracking targets')

    cv2.namedWindow("tracking")
    camera = cv2.VideoCapture('/home/haku/Yolo/darknet/data/test3.mp4')
    tracker = cv2.MultiTracker_create()
    init_once = False

    ok, image = camera.read()
    if not ok:
        print('Failed to read video')
        exit()

    bbox1 = cv2.selectROI('tracking', image)
    bbox2 = cv2.selectROI('tracking', image)
    bbox3 = cv2.selectROI('tracking', image)

    print(bbox1)

    while camera.isOpened():
        ok, image = camera.read()
        if not ok:
            print
            'no image to read'
            break

        if not init_once:
            ok = tracker.add(cv2.TrackerKCF_create(), image, (10, 10, 10, 10))
            ok = tracker.add(cv2.TrackerKCF_create(), image, bbox2)
            ok = tracker.add(cv2.TrackerKCF_create(), image, bbox3)
            init_once = True

        ok, boxes = tracker.update(image)

        for newbox in boxes:
            p1 = (int(newbox[0]), int(newbox[1]))
            p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
            cv2.rectangle(image, p1, p2, (200, 0, 0))

        cv2.imshow('tracking', image)
        cv2.waitKey(1)


def tracker_obj_3():
    net = load_net(b"/home/haku/Yolo/darknet/cfg/yolov3.cfg", b"/home/haku/Yolo/darknet/yolov3.weights", 0)
    meta = load_meta(b"/home/haku/Yolo/darknet/cfg/coco.data")
    camera = cv2.VideoCapture('/home/haku/Yolo/darknet/data/test3.mp4')

    cv2.namedWindow("tracking")
    tracker = cv2.MultiTracker_create()
    init_once = False

    while camera.isOpened():
        ok, image = camera.read()
        r = detect_vid(net, meta, image)
        my_person_list = Helper.get_list_person(r)

        if not init_once:
            for p in my_person_list:
                bottom_left_x, bottom_left_y, rcg_width, rcg_height = p.create_rectangle()
                tracker.add(cv2.TrackerMedianFlow_create(), image,
                            (bottom_left_x, bottom_left_y, rcg_width - bottom_left_x, rcg_height - bottom_left_y))

            init_once = True

        ok, boxes = tracker.update(image)

        for newbox in boxes:
            p1 = (int(newbox[0]), int(newbox[1]))
            p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
            cv2.rectangle(image, p1, p2, (200, 0, 0))

        cv2.imshow('tracking', image)
        cv2.waitKey(1)


if __name__ == "__main__":
    detect_from_video()
    # tracker_obj_3()
