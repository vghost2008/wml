import numpy as np
import sys
import copy

def _safe_persent(v0,v1):
    if v1==0:
        return 100.
    else:
        return v0*100./v1

class Accuracy:
    def __init__(self,topk=1,**kwargs):
        self.topk = topk
        self.all_correct = []
        self.accuracy = 100.0

    def __call__(self,output,target):
        '''
        output: [N0,...,Nn,num_classes] or [N0,...,Nn]
        target: [N0,...,Nn]
        '''
        if output.ndim==target.ndim:
            idx = output
            output = np.reshape(output,[-1])
            target = np.reshape(target,[-1])
        else:
            idx = np.argsort(output,axis=-1)
            idx = idx[...,-self.topk:]
            target = np.repeat(np.expand_dims(target,axis=-1),self.topk,axis=-1)
        correct = target==idx
        correct = np.reshape(correct,[-1])
        self.all_correct.append(correct)
    
    def num_examples(self):
        if len(self.all_correct)==0:
            return
        all_correct = np.concatenate(self.all_correct,axis=0)
        return all_correct.size


    def evaluate(self):
        self.accuracy = 100
        if len(self.all_correct)==0:
            return self.accuracy
        all_correct = np.concatenate(self.all_correct,axis=0)
        if all_correct.size == 0:
            return self.accuracy
        
        correct = float(np.sum(all_correct))

        self.accuracy = _safe_persent(correct,all_correct.size)

        return self.accuracy


    def show(self,name=""):
        sys.stdout.flush()
        self.evaluate()
        print(f"Test size {self.num_examples()}")
        print(f"accuracy={self.accuracy}")
        return self.accuracy

    def to_string(self):
        return f"Accuracy={self.accuracy:.2f}"

    def __repr__(self):
        return self.to_string()
    
    def value(self):
        return self.accuracy

class BAccuracy(Accuracy):
    def __init__(self,num_classes,**kwargs):
        '''
        二分类的正确率，最后一个类别为背景，其它类别为前景，只需要将背景或前景分正确即可
        '''
        self.bk_classes = num_classes-1
        super().__init__()

    def __call__(self,output,target):
        '''
        output: [N0,...,Nn,num_classes]
        target: [N0,...,Nn]
        '''
        idx = np.argsort(output,axis=-1)
        idx = idx[...,-1]
        labels = idx!=self.bk_classes
        target = target!=self.bk_classes
        labels = np.reshape(labels,[-1])
        target = np.reshape(target,[-1])
        return super().__call__(labels,target)

    def to_string(self):
        return f"BAccuracy={self.accuracy:.2f}"

class PrecisionAndRecall:
    def __init__(self,**kwargs):
        self.all_output = []
        self.all_target = []
        self.recall = 100.0
        self.precision = 100.0


    def __call__(self,output,target):
        '''
        output: [N0,...,Nn]
        target: [N0,...,Nn]
        '''
        self.all_output.append(np.reshape(output,[-1]))
        self.all_target.append(np.reshape(target,[-1]))
    
    def num_examples(self):
        if len(self.all_output)==0:
            return
        all_output = np.concatenate(self.all_output,axis=0)
        return all_output.size


    def evaluate(self):
        self.recall = 100
        self.precision = 100
        if len(self.all_output)==0:
            return self.precision,self.recall
        all_output = np.concatenate(self.all_output,axis=0)
        if all_output.size == 0:
            return self.precision,self.recall
        
        all_target = np.concatenate(self.all_target,axis=0)
        tmp_mask = all_output==all_target
        correct = np.sum(all_output[tmp_mask].astype(np.float32))

        tp_fp = np.sum(all_output.astype(np.float32))
        tp_fn = np.sum(all_target.astype(np.float32))

        self.precision = _safe_persent(correct,tp_fp)
        self.recall = _safe_persent(correct,tp_fn)

        return self.precision,self.recall


    def show(self,name=""):
        sys.stdout.flush()
        self.evaluate()
        print(f"Test size {self.num_examples()}")
        print(self.to_string())

    def to_string(self):
        return f"P={self.precision:.2f}, R={self.recall:.2f}"

    def __repr__(self):
        return self.to_string()
    
    def value(self):
        return _safe_persent(self.precision*self.recall,self.precision+self.recall) #F1

class BPrecisionAndRecall(PrecisionAndRecall):
    def __init__(self,num_classes,**kwargs):
        '''
        二分类的精度与召回，最后一个类别为背景，其它类别为前景，只需要将背景或前景分正确即可
        '''
        self.bk_classes = num_classes-1
        super().__init__(**kwargs)


    def __call__(self,output,target):
        '''
        output: [N0,...,Nn,num_classes]
        target: [N0,...,Nn]
        '''
        idx = np.argsort(output,axis=-1)
        idx = idx[...,-1]
        labels = idx!=self.bk_classes
        target = target!=self.bk_classes
        labels = np.reshape(labels,[-1])
        target = np.reshape(target,[-1])
        return super().__call__(labels,target)

    def to_string(self):
        return f"BP={self.precision:.2f}, BR={self.recall:.2f}"

class ConfusionMatrix:
    def __init__(self,num_classes=-1,**kwargs):
        self.all_target = []
        self.all_pred = []
        self.accuracy = 100.0
        self.num_classes = num_classes
        self.cm = []

    def __call__(self,output,target):
        '''
        output: [N0,...,Nn,num_classes]
        target: [N0,...,Nn]
        '''
        if self.num_classes<=0:
            self.num_classes = output.shape[-1]
        idx = np.argsort(output,axis=-1)
        pred = idx[...,-1]
        self.all_pred.append(copy.deepcopy(np.reshape(pred,[-1])))
        self.all_target.append(copy.deepcopy(np.reshape(target,[-1])))
    
    def num_examples(self):
        if len(self.all_pred)==0:
            return
        all_pred= np.concatenate(self.all_pred,axis=0)
        return all_pred.size


    def evaluate(self):
        if len(self.all_pred)==0:
            return ""

        cm = np.zeros([self.num_classes,self.num_classes],dtype=np.int32)
        all_pred= np.concatenate(self.all_pred,axis=0)
        all_target = np.concatenate(self.all_target,axis=0)

        for p,t in zip(all_pred,all_target):
            cm[t,p] = cm[t,p]+1
        
        self.cm = cm

        return cm



    def show(self,name=""):
        sys.stdout.flush()
        self.evaluate()
        print(self.to_string())
        return self.accuracy

    def to_string(self,blod=True):
        res = "\n"
        for i in range(self.num_classes):
            line = ""
            for j in range(self.num_classes):
                if blod and i==j:
                    #line += f"\033[1m{self.cm[i,j]:<5}\033[0m, "
                    line += f"{self.cm[i,j]:<4}*, "
                else:
                    line += f"{self.cm[i,j]:<5}, "
            res += line+"\n"
        return res

    def __repr__(self):
        return self.to_string()
    
    def value(self):
        return self.cm

class ClassesWiseModelPerformace:
    def __init__(self,num_classes,classes_begin_value=0,model_type=PrecisionAndRecall,model_args={},label_trans=None,
                  name=None,
                 **kwargs):
        self.num_classes = num_classes
        self.clases_begin_value = classes_begin_value
        model_args['classes_begin_value'] = classes_begin_value
        self.data = []
        for i in range(self.num_classes):
            self.data.append(model_type(num_classes=num_classes,**model_args))
        self.label_trans = label_trans
        self.have_data = np.zeros([num_classes],dtype=np.bool)
        self.accuracy = Accuracy(topk=1)
        self.name = name

    @staticmethod
    def select_labels(labels,target,classes):
        if len(labels) == 0:
            return np.array([],dtype=np.int32),np.array([],dtype=np.int32)
        if not isinstance(labels,np.ndarray):
            labels = np.array(labels)
        mask0 = np.equal(labels,classes)
        mask1 = np.equal(target,classes)
        mask = np.logical_or(mask0,mask1)
        nlabels = (labels[mask]==classes).astype(np.int32)
        ntarget = (target[mask]==classes).astype(np.int32)
        return nlabels,ntarget

    def __call__(self,output,target):
        '''
        output: [N0,...,Nn,num_classes]
        target: [N0,...,Nn]
        '''
        self.accuracy(output,target)
        idx = np.argsort(output,axis=-1)
        labels = np.reshape(idx[...,-1:],[-1])
        target = np.reshape(target,[-1])

        if self.label_trans is not None:
            labels = self.label_trans(labels)
            target = self.label_trans(target)
        
        for i in range(self.num_classes):
            classes = i+self.clases_begin_value
            clabels,ctarget = self.select_labels(labels,target,classes)
            self.have_data[i] = True
            self.data[i](clabels,ctarget)

        self._current_info = ""


    def show(self,name=""):
        sys.stdout.flush()
        for i in range(self.num_classes):
            if not self.have_data[i]:
                continue
            classes = i+self.clases_begin_value
            print(f"Classes:{classes}")
            try:
                self.data[i].show(name=name)
            except:
                print("N.A.")
                pass

    def evaluate(self):
        for d in self.data:
            d.evaluate()
        self.accuracy.evaluate()

    def to_string(self):
        res = ";".join([str(idx)+": "+d.to_string() for idx,d in enumerate(self.data)]+[self.accuracy.to_string()])
        if self.name is not None:
            res = f"{self.name}: "+res
        return res

    def __repr__(self):
        return self.to_string()

class ComposeMetrics:
    def __init__(self,*args,**kwargs):
        self.metrics = list(args)+list(kwargs.values())

    def __call__(self, *args,**kwargs):
        [m(*args,**kwargs) for m in self.metrics]

    def evaluate(self):
        [m.evaluate() for m in self.metrics]

    def show(self,name=""):
        [m.show(name=name) for m in self.metrics]

    def to_string(self):
        return ";".join([m.to_string() for m in self.metrics])

    def value(self):
        return self.metrics[0].value()