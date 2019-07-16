COCO = [0.2695, 0.05400000000000003, 0.22050000000000006, 0.0045, 0.18550000000000005, 0.20600000000000002, 0.05900000000000003]
CALTECH = [0.001, 0.9909999999999999, 0.0, 0.0, 0.0, 0.003, 0.0]
PASCAL = [0.22150000000000009, 0.03600000000000001, 0.288, 0.004999999999999999, 0.18650000000000003, 0.196, 0.06700000000000003]
CAM2 = [0.001, 0.0, 0.0005, 0.9884999999999999, 0.0015, 0.0015, 0.0005]
INRIA = [0.1825, 0.057000000000000016, 0.167, 0.0035000000000000005, 0.32899999999999996, 0.17050000000000004, 0.09050000000000001]
SUN = [0.20350000000000001, 0.05750000000000003, 0.19650000000000006, 0.006999999999999999, 0.182, 0.263, 0.09150000000000003]
KITTI = [0.018500000000000006, 0.0055, 0.0175, 0.0, 0.019, 0.024000000000000004, 0.9140000000000003]

COCO_rounded = [ round(elem, 2) for elem in COCO ]
COCO_rounded = [i * 100 for i in COCO_rounded]
print(' '.join([str(int(x)) for x in COCO_rounded]))
#print(sum(COCO_rounded))
CALTECH_rounded = [ round(elem, 2) for elem in CALTECH ]
CALTECH_rounded = [i * 100 for i in CALTECH_rounded]
print(' '.join([str(int(x)) for x in CALTECH_rounded]))
#print(sum(CALTECH_rounded))
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
