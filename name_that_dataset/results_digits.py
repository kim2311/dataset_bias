USPS = [0.9681250000000001, 0.031875, 0.0, 0.0]
MNIST = [0.041875, 0.9581249999999999, 0.0, 0.0]
MNIST_M = [0.0037500000000000003, 0.001875, 0.9687500000000002, 0.023125000000000007]
SVHN = [0.0, 0.00125, 0.028125000000000008, 0.9687500000000001]

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
