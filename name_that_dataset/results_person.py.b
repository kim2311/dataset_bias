COCO = [0.38166666666666665, 0.315, 0.2691666666666667, 0.03333333333333333]
VOC = [0.31916666666666665, 0.3941666666666667, 0.2575, 0.029166666666666664]
SUN = [0.27166666666666667, 0.2658333333333333, 0.40833333333333327, 0.055833333333333346]
KITTI = [0.015833333333333335, 0.016666666666666666, 0.03666666666666666, 0.9316666666666665] 

COCO_rounded = [ round(elem, 2) for elem in COCO ]
print(' '.join(map(str, COCO_rounded)))
VOC_rounded = [ round(elem, 2) for elem in VOC ]
print(' '.join(map(str, VOC_rounded)))
SUN_rounded = [ round(elem, 2) for elem in SUN ]
print(' '.join(map(str, SUN_rounded)))
KITTI_rounded = [ round(elem, 2) for elem in KITTI ]
print(' '.join(map(str, KITTI_rounded)))
