import time
import torch
import torchvision.models as models
from os import environ

from model import darknet53

def speed(model, name):
    with torch.no_grad():
        model.eval()

        input = torch.rand(1,3,224, 224).cuda()

        model(input)

        avg_time = 0

        for i in range(0, 10):
            torch.cuda.synchronize()
            t2 = time.time()

            model(input)

            torch.cuda.synchronize()
            t3 = time.time()

            avg_time += t3 - t2

        avg_time /= 10.0

        print('%12s : %3.3f ms' % (name, avg_time * 1000))


if __name__ == '__main__':
    environ["CUDA_LAUNCH_BLOCKING"] = "1"
    torch.backends.cudnn.benchmark = True

    resnet50 = models.resnet50().cuda()
    resnet101 = models.resnet101().cuda()
    resnet152 = models.resnet152().cuda()
    densenet121 = models.densenet121().cuda()
    darknet = darknet53(1000).cuda()

    speed(resnet50, 'resnet50')
    speed(resnet101, 'resnet101')
    speed(resnet152, 'resnet152')
    speed(densenet121, 'densenet121')
    speed(darknet, 'darknet53')
