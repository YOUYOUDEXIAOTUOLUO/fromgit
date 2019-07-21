import cv2
from PIL import Image
from PIL import ImageGrab
import numpy

video = cv2.VideoCapture("C:/Users/gaoji/Desktop/speak2/VID_20190616_153834.mp4")
fps = video.get(cv2.CAP_PROP_FPS)
frameCount = video.get(cv2.CAP_PROP_FRAME_COUNT)
size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
inputing = 1


while inputing:

    success, frame = video.read()
    index = 1
    target_name = input("嵌入图像文件名：")
    if target_name == 'none':
        break
    roi = Image.open("C:/Users/gaoji/Desktop/speak2/"+target_name)
    w, h = roi.size
    roi.thumbnail((w/2.1, h/2.1))
    target_start = (int(input("嵌入开始时间：")) - 1)*fps
    target_end = int(input("嵌入结束时间："))*fps

    videoWriter = cv2.VideoWriter("C:/Users/gaoji/Desktop/speak2/test.MP4", cv2.VideoWriter_fourcc(*'MP4V'), fps, size)
    while success:
        if target_start <= index <= target_end:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image.paste(roi, (5, 160))
            frame = cv2.cvtColor(numpy.asarray(image), cv2.COLOR_RGB2BGR)
        else:
            target_name = input("嵌入图像文件名：")
            if target_name == 'none':
                success = 0
                inputing = 0
            else:
                roi = Image.open("C:/Users/gaoji/Desktop/speak2/" + target_name)
                w, h = roi.size
                roi.thumbnail((w / 2.1, h / 2.1))
                target_start = (int(input("嵌入开始时间：")) - 1) * fps
                target_end = int(input("嵌入结束时间：")) * fps
            continue

        videoWriter.write(frame)
        success, frame = video.read()
        index += 1

    video.release()



