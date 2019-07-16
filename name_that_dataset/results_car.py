COCO = [0.3192307692307692, 0.1253846153846154, 0.2330769230769231, 0.22538461538461543, 0.09769230769230769]
IMAGENET = [0.08461538461538462, 0.6830769230769231, 0.13769230769230767, 0.07615384615384616, 0.01769230769230769]
PASCAL = [0.21461538461538465, 0.2738461538461539, 0.24153846153846154, 0.19538461538461535, 0.07384615384615385]
SUN = [0.21384615384615385, 0.10076923076923079, 0.1623076923076923, 0.4446153846153847, 0.07846153846153844]
KITTI = [0.016923076923076923, 0.006923076923076922, 0.012307692307692308, 0.015384615384615384, 0.9515384615384614]

COCO_rounded = [ round(elem, 2) for elem in COCO ]
COCO_rounded = [i * 100 for i in COCO_rounded]
print(' '.join([str(int(x)) for x in COCO_rounded]))
#print(sum(COCO_rounded))
IMAGENET_rounded = [ round(elem, 2) for elem in IMAGENET ]
IMAGENET_rounded = [i * 100 for i in IMAGENET_rounded]
print(' '.join([str(int(x)) for x in IMAGENET_rounded]))
#print(sum(IMAGENET_rounded))
PASCAL_rounded = [ round(elem, 2) for elem in PASCAL ]
PASCAL_rounded = [i * 100 for i in PASCAL_rounded]
print(' '.join([str(int(x)) for x in PASCAL_rounded]))
#print(sum(PASCAL_rounded))
SUN_rounded = [ round(elem, 2) for elem in SUN ]
SUN_rounded = [i * 100 for i in SUN_rounded]
print(' '.join([str(int(x)) for x in SUN_rounded]))
#print(sum(SUN_rounded))
KITTI_rounded = [ round(elem, 2) for elem in KITTI ]
KITTI_rounded = [i * 100 for i in KITTI_rounded]
print(' '.join([str(int(x)) for x in KITTI_rounded]))
#print(sum(KITTI_rounded))
