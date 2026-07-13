import numpy as np

# MLP
out = np.load('/tmp/test_mlp/logits.npy')
gold = np.load('/home/mig20/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/testdata/c35/mlp_v1/golden/logits.npy')
lab = np.load('/home/mig20/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/testdata/c35/mlp_v1/labels.npy')
print('MLP:')
print(f'  Our accuracy:    {(out.argmax(1)==lab).mean():.4f}')
print(f'  Golden accuracy: {(gold.argmax(1)==lab).mean():.4f}')
print(f'  Argmax diff:     {(out.argmax(1)!=gold.argmax(1)).sum()} / {len(lab)}')

# ResNet
out2 = np.load('/tmp/test_resnet/logits.npy')
gold2 = np.load('/home/mig20/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/testdata/c35/resnet_v1/golden/logits.npy')
lab2 = np.load('/home/mig20/Agentic4SystemSummerSchoolContest/Track-C/C3-scheduler/testcases/release_to_competitors/testdata/c35/resnet_v1/labels.npy')
print('ResNet:')
print(f'  Our accuracy:    {(out2.argmax(1)==lab2).mean():.4f}')
print(f'  Golden accuracy: {(gold2.argmax(1)==lab2).mean():.4f}')
print(f'  Argmax diff:     {(out2.argmax(1)!=gold2.argmax(1)).sum()} / {len(lab2)}')
