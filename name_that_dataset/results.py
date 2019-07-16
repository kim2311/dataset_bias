COCO = [0.20124999999999998, 0.04, 0.17875, 0.17124999999999999, 0.02375, 0.15125, 0.16124999999999998, 0.031249999999999997, 0.00625, 0.0025, 0.02, 0.01625]
CALTECH = [0.0025, 0.9800000000000002, 0.0, 0.0, 0.0, 0.00125, 0.005, 0.0, 0.0, 0.0, 0.0, 0.0]
IMAGENET = [0.16499999999999998, 0.03874999999999999, 0.24125, 0.17374999999999996, 0.01875, 0.125, 0.12875, 0.030000000000000002, 0.012499999999999999, 0.015000000000000001, 0.02625, 0.027500000000000004]
PASCAL = [0.16999999999999998, 0.0325, 0.1775, 0.215, 0.02375, 0.13875, 0.15125, 0.035, 0.007500000000000001, 0.005, 0.02375, 0.02125]
CAM2 = [0.00125, 0.00, 0.0, 0.00125, 0.9887500000000001, 0.00125, 0.00125, 0.00125, 0.0, 0.0, 0.0, 0.0]
INRIA = [0.14375, 0.03875, 0.13375, 0.15375, 0.013749999999999998, 0.28250000000000003, 0.13625, 0.047499999999999994, 0.005, 0.0, 0.015, 0.0225]
SUN = [0.17124999999999999, 0.03875, 0.1325, 0.14625000000000002, 0.028749999999999998, 0.155, 0.23374999999999999, 0.04, 0.005, 0.00375, 0.00875, 0.035]
KITTI = [0.030000000000000002, 0.01, 0.02375, 0.0275, 0.0075, 0.0325, 0.026250000000000002, 0.8362499999999999, 0.0, 0.0, 0.00125, 0.00375]
USPS = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9674999999999999, 0.02375, 0.0, 0.0]
MNIST = [0.0, 0.0, 0.0, 0.00125, 0.0, 0.0, 0.0, 0.0, 0.0325, 0.9625, 0.0, 0.0]
MNIST_M = [0.01875, 0.00125, 0.0225, 0.024999999999999994, 0.008749999999999999, 0.02, 0.013750000000000002, 0.0, 0.0025, 0.005, 0.82875, 0.0525]
SVHN = [0.00125, 0.00125, 0.005, 0.00375, 0.0, 0.0025, 0.0025, 0.0, 0.0, 0.00125, 0.012499999999999999, 0.9662499999999998]

COCO_rounded = [ round(elem, 2) for elem in COCO ]
COCO_rounded = [i * 100 for i in COCO_rounded]
print(' '.join([str(int(x)) for x in COCO_rounded]))
#print(sum(COCO_rounded))
CALTECH_rounded = [ round(elem, 2) for elem in CALTECH ]
CALTECH_rounded = [i * 100 for i in CALTECH_rounded]
print(' '.join([str(int(x)) for x in CALTECH_rounded]))
#print(sum(CALTECH_rounded))
IMAGENET_rounded = [ round(elem, 2) for elem in IMAGENET ]
IMAGENET_rounded = [i * 100 for i in IMAGENET_rounded]
print(' '.join([str(int(x)) for x in IMAGENET_rounded]))
#print(sum(IMAGENET_rounded))
PASCAL_rounded = [ round(elem, 2) for elem in PASCAL ]
PASCAL_rounded = [i * 100 for i in PASCAL_rounded]
print(' '.join([str(int(x)) for x in PASCAL_rounded]))
#print(sum(PASCAL_rounded))
CAM2_rounded = [ round(elem, 2) for elem in CAM2 ]
CAM2_rounded = [i * 100 for i in CAM2_rounded]
print(' '.join([str(int(x)) for x in CAM2_rounded]))
#print(sum(CAM2_rounded))
INRIA_rounded = [ round(elem, 2) for elem in INRIA ]
INRIA_rounded = [i * 100 for i in INRIA_rounded]
print(' '.join([str(int(x)) for x in INRIA_rounded]))
#print(sum(INRIA_rounded))
SUN_rounded = [ round(elem, 2) for elem in SUN ]
SUN_rounded = [i * 100 for i in SUN_rounded]
print(' '.join([str(int(x)) for x in SUN_rounded]))
#print(sum(SUN_rounded))
KITTI_rounded = [ round(elem, 2) for elem in KITTI ]
KITTI_rounded = [i * 100 for i in KITTI_rounded]
print(' '.join([str(int(x)) for x in KITTI_rounded]))
#print(sum(KITTI_rounded))
USPS_rounded = [ round(elem, 2) for elem in USPS ]
USPS_rounded = [i * 100 for i in USPS_rounded]
print(' '.join([str(int(x)) for x in USPS_rounded]))
#print(sum(USPS_rounded))
MNIST_rounded = [ round(elem, 2) for elem in MNIST ]
MNIST_rounded = [i * 100 for i in MNIST_rounded]
print(' '.join([str(int(x)) for x in MNIST_rounded]))
#print(sum(MNIST_rounded))
MNIST_M_rounded = [ round(elem, 2) for elem in MNIST_M ]
MNIST_M_rounded = [i * 100 for i in MNIST_M_rounded]
print(' '.join([str(int(x)) for x in MNIST_M_rounded]))
#print(sum(MNIST_M_rounded))
SVHN_rounded = [ round(elem, 2) for elem in SVHN ]
SVHN_rounded = [i * 100 for i in SVHN_rounded]
print(' '.join([str(int(x)) for x in SVHN_rounded]))
#print(sum(SVHN_rounded))
