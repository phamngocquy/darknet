import multiprocessing as mp
import time
import cv2
from dlib import correlation_tracker, rectangle
import get_points


class Track(correlation_tracker):
    def __init__(self):
        correlation_tracker.__init__(self)

    def start(self, img, det):
        correlation_tracker.start_track(self, img, det)

    def renew(self, img):
        return correlation_tracker.update(self, img)

    def loc(self):
        return correlation_tracker.get_position(self)


def update(tracker, image):
    tracker.renew(image)

def main():
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    if not cam.isOpened():
        print("Couldn't open video stream")
        exit()
    while True:
        retval, img = cam.read()
        if not retval:
            print("Couldn't read frame from device")
            exit()
        key = cv2.waitKey(1)
        if key == 32:
            break
        if key == 27:
            cv2.destroyAllWindows()
            print("Exited program")
            exit()
        cv2.imshow("Image", img)
    cv2.destroyWindow("Image")
    points = get_points.run(img, multi=True)
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Image", img)

    tracker = [Track() for _ in range(len(points))]
    [tracker[i].start(img, rectangle(*rect)) for i, rect in enumerate(points)]
    while True:
        # pool = mp.Pool(mp.cpu_count())
        procs = []
        retval, img = cam.read()
        if not retval:
            print("Couldn't read frame from device")
        start = time.time()
        for i in range(len(tracker)):
            # tracker[i].renew(img)
            # update(tracker[i], img)
            p = mp.Process(target=update, args=(tracker[i], img))
            p.start()
            procs.append(p)

        for p in procs:
            p.join()

        print("Time taken (tracker update):{0:.5f}".format(time.time() - start))
        for i in range(len(tracker)):
            rect = tracker[i].get_position()
            rect = tracker[i].loc()
            pt1 = (int(rect.left()), int(rect.top()))
            pt2 = (int(rect.right()), int(rect.bottom()))
            cv2.rectangle(img, pt1, pt2, (255, 255, 255), 3)
        cv2.imshow("Image", img)
        if cv2.waitKey(1) == 27:
            break


if __name__ == "__main__":
    main()