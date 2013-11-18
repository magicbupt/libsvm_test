
import sys
sys.path.append('../libsvm_python')

#from svm import *
from svmutil import *
print sys.platform
print "this is libsvm_python test project"
y, x = [1,-1], [[1,0,1], [-1,0,-1]]
y, x = [1,-1], [{1:1, 3:1}, {1:-1,3:-1}]
prob  = svm_problem(y, x)
param = svm_parameter('-t 0 -c 4 -b 1')



#test is really ok?
m = svm_train(prob, param)
svm_save_model('../data/heart_scale.svm.model', m)
m = svm_load_model('../data/heart_scale.svm.model')
p_label, p_acc, p_val = svm_predict(y, x, m, '-b 0')
ACC, MSE, SCC = evaluations(y, p_label)