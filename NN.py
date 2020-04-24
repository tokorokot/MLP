#toDo List:
#-внимательно все проверить
#-реализовать регуляризацию
#-подумать над тем, чтобы вынести все преведения к типу ndarray из класса наружу для оптимизации




import numpy as np
import matplotlib.pyplot as plt
import time 
import pickle

class NNet_exc(Exception): pass

def act_func(x,mode='sigmoid'): #функция активации
    x=np.array(x)
    mode=mode.split(' ')
    if len(mode)==1:
        alph=1.
    else:
        alph=float(mode[1])
    
    if mode[0]=='sigmoid':
        #print(mode, alph)
        return (1/(1+np.exp(-alph*x)))
        
    elif mode[0]=='tanh':
        #print(mode, alph)
        return ((1-np.exp(-alph*x))/(1+np.exp(-alph*x)))
        
    elif mode[0]=='ReLu':
        #print(mode, alph)
        return np.maximum(0,x)
        #Для ReLu нужно очень медленное обучения, а то "умрет"
        
    elif mode[0]=='leakyReLu':
        #print(mode, alph)
        y=x.astype(float)
        for a in range(0,len(y)):
            
            if y[a]<=0: y[a]=(y[a]*0.01)
        return y
        
        
    
def loss_func(out,out_exp): #функция потерь
    return np.sum(0.5*np.square(out_exp-out))/len(out)
    
def act_func_deriv(x,mode='sigmoid'): #производная функции активации
    x=np.array(x)
    mode_=mode
    mode=mode.split(' ')
    if len(mode)==1:
        alph=1.
    else:
        alph=float(mode[1])
        
    
    if mode[0]=='sigmoid':
        #print(' ',mode, alph)
        return (alph*act_func(x,mode_)*(1-act_func(x,mode_)))
    elif mode[0]=='tanh':
        #print(' ',mode, alph)
        return (0.5*alph*(1-act_func(x,mode_)*act_func(x,mode_)))
    elif mode[0]=='ReLu':
        #print(' ',mode, alph)
        y=act_func(x,'ReLu')
        for a in range(0,len(y)):
            if y[a]!=0: y[a]=1 
        return y
        
    elif mode[0]=='leakyReLu':
        #print(mode, alph)
        y=x.astype(float)
        for a in range(0,len(y)):
            if y[a]<=0: y[a]=0.01
            else: y[a]=1
        return y


def NNet_load(file_path):#обертка для закрузчика обученной сети через pickle
    file=open(file_path,'rb')
    b=pickle.load(file)
    print('Loaded network:')
    b.show_conf()
    



class TrainInfo:
    def __init__(self):
        self.epoch=0 #текущая эпоха обучения
        self.iter_lst=[] # список векторов итераций для каждой эпохи
        self.loss_lst=[] # список векторов ошибок для каждой эпохи 
        self.time_lst=[] # список времени, затраченного на эпоху
        
        plt.ion()
          
    
    def plot(self,mode='loss_epoch', epoch=0):
        fig, ax = plt.subplots()
                        
        if mode=='loss':
            
            x=self.iter_lst[epoch]
            y=self.loss_lst[epoch]
            ax.set_ylabel('loss')
            ax.set_xlabel('iteration')  
            ax.set_title("loss vs iteration on epoch %s"%(epoch))  
        elif mode== 'time':
            x=np.linspace(0,self.epoch-1,self.epoch)
            
            y=self.time_lst
            
            ax.set_ylabel('time')
            ax.set_xlabel('epoch')  
            ax.set_title("time vs epoch")
        elif mode=='loss_epoch':
            x=np.linspace(0,self.epoch-1,self.epoch)
            
            y=[]
            for a in self.loss_lst:
                a=np.sum(a)/len(a)
                y.append(a)
            ax.set_ylabel('average loss')
            ax.set_xlabel('epoch')  
            ax.set_title("loss vs epoch")
            
            
        ax.plot(x,y)    
        
        plt.show()
        
    def clear(self):
        self.epoch=0 
        self.iter_lst.clear()
        self.loss_lst.clear() 
        self.time_lst.clear()





class NNet:
    
    def __init__ (self, inp_size, out_size, mid_conf=[], mode='sigmoid'):
        
        #проверка параметра для скрытых слоев. Если int, то конвертируется в list
        if type(mid_conf)==int:
            mid_conf=list([mid_conf])
            
        self.layers=len(mid_conf)+1 #слоев в сети (однослойный - состоящий только из входа и выхода). ОЧЕНЬ ВАЖНЫЙ ПАРАМЕТР, использующийся далее для проходке и итерированию по сети
        
        self.w_matrx=[] #список из матриц весов между слоями
        self.out_val_matrx=[] #список из столбцов значений на выходах узлов сети (вклюяая входные значения)
        self.inp_val_matrx=[] #список из столбцов значений на входе в узлы (до активации) (включая входные значения)
        self.del_w_matrx=[] #список из матриц корректировки весов. Нужен для учета предыдуще корректировки
                
        self.bias_matrx=[] #список столбцов сдвигов
        self.del_bias_matrx=[] #список из матриц корректировки весов смещений. Нужен для учета предыдуще корректировки
        
        self.conf=[inp_size]+mid_conf+[out_size] #спсок, сответствующий конфигурации сети (кол-во узлов на каждом слое)
        
        self.inf=TrainInfo() #параметр для хранения информации об обучении
        
        self.bias=1 #отладочная переменная включения и отключения сдвиговых узлов
        self.mode=mode # функции активации
        
        for a in range (0, self.layers):
            arr=np.random.rand(self.conf[a+1], self.conf[a]) #инициализация весов произвольными значениями
            self.w_matrx.append(arr)
            arr=np.zeros((self.conf[a+1], self.conf[a])) #инициализация матриц корректировки весов нулями
            self.del_w_matrx.append(arr)
            arr=np.random.rand(self.conf[a+1], 1) #инициализация весов узлов смещения произвольными значениями
            self.bias_matrx.append(arr)
            arr=np.zeros((self.conf[a+1], 1)) #инициализация матриц корректировки весов смещений нулями
            self.del_bias_matrx.append(arr)
            
          
    
    def fwd_calc(self,inp_v): #проход сети в прямом направлении
        if (len(inp_v) != self.conf[0]):
            raise NNet_exc('input data does not match Network')
       
        
        inp_vals=np.transpose(np.array([inp_v]))
        
        
       
        self.out_val_matrx=[inp_vals]
        
        self.inp_val_matrx=[inp_vals]
        
        for a in range (0, self.layers):
            inp_field=self.w_matrx[a] @ self.out_val_matrx[a] + self.bias*self.bias_matrx[a]
            
            out_field=act_func(inp_field,self.mode)
            self.inp_val_matrx.append(inp_field)
            self.out_val_matrx.append(out_field)
            
        
        return out_field
        
    def backprop(self, exp_v, nu=0.5, alph=0): #обратное распространение ошибки, nu - скорость обучения, alph - момент
        if type(exp_v)==int:
            exp_v=list([exp_v])
        if (len(exp_v) != self.conf[-1]):
            raise NNet_exc('output data does not match Network')
        if (len(self.inp_val_matrx)==0):
            raise NNet_exc('Weights could not be backpropogated without at least one iter of forward propogation')
        
        #со смещениями работаю отдельно от основной матрицы весов
        
        exp_vals=np.transpose(np.array([exp_v]))
        
        delta_matrx=[]  #список, состоящий из столбцов локальных градиентов сети. Изначально пустой. Далее в начало списка будут вставляться столбц локальных градинтов слоев сети
                        #начиная с последнего слоя. Это нужно для удобства распространения ошибки.
        #столбец для последнего слоя
        #print(self.inp_val_matrx[self.layers-1])
        #print(self.out_val_matrx[self.layers-1])
        dj=act_func_deriv(self.inp_val_matrx[self.layers],self.mode)*(exp_vals-self.out_val_matrx[self.layers])
        delta_matrx.insert(0,dj)
        #остальные столбцы
        for a in range(self.layers-1,0,-1):
            dj=act_func_deriv(self.inp_val_matrx[a],self.mode)*(self.w_matrx[a].T @ delta_matrx[0])
            delta_matrx.insert(0,dj)
        
        #print(delta_matrx)
        
        for a in range(0,self.layers):
            self.del_w_matrx[a]=alph*self.del_w_matrx[a] + nu*(delta_matrx[a]@self.out_val_matrx[a].T)
            self.del_bias_matrx[a]=alph*self.del_bias_matrx[a] + nu*(delta_matrx[a]*self.bias) #если смещения отключены, то корректировки не будет
            
            self.w_matrx[a]=self.w_matrx[a]+self.del_w_matrx[a]
            self.bias_matrx[a]=self.bias_matrx[a]+self.del_bias_matrx[a]
        
        #print(self.del_w_matrx)
        
    def train(self,inp_lst,out_lst,nu=0.5, alph=0, mode='release'): #mode может принимать значения release, silent, debug
                                                                    #release - выводится инфа о текущей эпехе и ошибке на каждой итерации
                                                                    #debug - к release  добавляется информация о весах и input/output узлов сети
                                                                    #silent - не выводится никакая информация
        if len(inp_lst)!=len(out_lst):
            raise NNet_exc('train error. In/out sizzes mismatch')
        
        loss=[]
        iter=np.linspace(0,len(inp_lst)-1,len(inp_lst))
        start=time.time()
        
        if mode!='silent':#"тихий" режим
            print(' Epoch: %d'%(self.inf.epoch))
        
        for a in range(0,len(inp_lst)):
            if mode=='debug': #вывод отладочной инфы, если режим установлен в debug
                self.show_w() 
            
            rez=self.fwd_calc(inp_lst[a])
            loss.append(loss_func(rez,out_lst[a]))            
            self.backprop(out_lst[a],nu, alph)
            
            if mode=='debug':
                self.show_inp()  
                self.show_out()
                
            if mode=='debug':
                self.show_del_w() 
                
            if mode!='silent':    
                print('   iteration: %d. Loss: %f'%(a,loss[a]))
                
        
        self.inf.time_lst.append(time.time()-start)
        self.inf.iter_lst.append(iter)
        self.inf.loss_lst.append(loss)
        
        self.inf.epoch+=1
        
        
        
        
    def show_conf(self): #вывод конфигурации сети
        print('Network mode: ', self.mode)
        print('configuration of NN layers: ', self.conf)
        
        if (len(self.conf)==2):
            print('%d input parameters,\n%d output parameters' %(self.conf[0],self.conf[-1]))  
                       
        else:   
            print('%d input parameters,\n%d output parameters,\n%s are number of nodes in %d hidden layer' %(self.conf[0],self.conf[-1],self.conf[1:-1],len(self.conf[1:-1])))
        
        if self.bias==0:
            print('Calculation without bias matrix')
            
        print('- - - - - - - - - - - - - - - - -')
        print('current epoch is ', self.inf.epoch)
        print('- - - - - - - - - - - - - - - - -') 
        print('- - - - - - - - - - - - - - - - -') 
        
           
            
    def show_w(self): #вывод весовых матриц сети
        print('Weight matrices are:')
        for a in range (0, len(self.w_matrx)):
            print(self.w_matrx[a])
            
            if self.bias==1:
                print('bias matrix:')
                print(self.bias_matrx[a])
            print('- - - - - - - - - - - - - - - - -')
        
        print('- - - - - - - - - - - - - - - - -') 
    
    def show_del_w(self): #вывод корректировок весовых матриц сети
        print('Weight matrices corrections are:')
        for a in range (0, len(self.del_w_matrx)):
            print(self.del_w_matrx[a])
            
            if self.bias==1:
                print('bias correction matrix:')
                print(self.del_bias_matrx[a])
            print('- - - - - - - - - - - - - - - - -')
            
        print('- - - - - - - - - - - - - - - - -') 
            
    def show_inp(self): #вывод входных матриц нейронов  сети
        print('Input matrices are:')
        if len(self.inp_val_matrx)==0:
            print('empty')
            return
        for a in range (0, len(self.inp_val_matrx)):
            print(self.inp_val_matrx[a])
            print('- - - - - - - - - - - - - - - - -')    
        
        print('- - - - - - - - - - - - - - - - -') 
            
    def show_out(self): #вывод входных матриц нейронов  сети
        print('Output matrices are:')
        if len(self.out_val_matrx)==0:
            print('empty')
            return
        for a in range (0, len(self.inp_val_matrx)):
            print(self.out_val_matrx[a])
            print('- - - - - - - - - - - - - - - - -')   
        print('- - - - - - - - - - - - - - - - -') 
        
            
    
    def save(self):
        file=open('sav.s','wb')       
        pickle.dump(self,file)
        file.close()
        return file.name
        
    def reset(self):
               
        self.out_val_matrx.clear()
        self.inp_val_matrx.clear()
        self.w_matrx.clear()
        self.bias_matrx.clear()
        self.del_w_matrx.clear()
        self.del_bias_matrx.clear()
        
        self.bias=1 
        
        self.inf.clear()
        
        for a in range (0, self.layers):
            arr=np.random.rand(self.conf[a+1], self.conf[a]) #инициализация весов произвольными значениями
            self.w_matrx.append(arr)
            arr=np.zeros((self.conf[a+1], self.conf[a])) #инициализация матриц корректировки весов нулями
            self.del_w_matrx.append(arr)
            arr=np.random.rand(self.conf[a+1], 1) #инициализация весов узлов смещения произвольными значениями
            self.bias_matrx.append(arr)
            arr=np.zeros((self.conf[a+1], 1)) #инициализация матриц корректировки весов смещений нулями
            self.del_bias_matrx.append(arr)
        #print(self.del_w_matrx) 
        
    
            
    
if __name__=='__main__':
        

    a=NNet(2,1,[3,3],mode='ReLu')
    
    x=[[0,0],[0,0],[0,0],[0,1],[1,0],[1,1]]
    
    y=(0,0,0,1,1,1)
    a.show_conf()
    a.train(x,y)
    a.train(x,y)
    a.train(x,y)
    #a.inf.plot(mode='time')
    #for n in range(0,a.inf.epoch):
    #    a.inf.plot('loss',n) 
    
    #a.inf.plot('loss_epoch')
    a.show_w()
    f=a.save()
    file=open(f,'rb')
    b=pickle.load(file)
    
    b.show_conf()
    b.show_w()
    

    #b.
    #input()



