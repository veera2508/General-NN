class gnn:
    def __init__(self, dims, activations, epochs, lr):
        self.dims = dims
        self.activations = activations
        self.epochs = epochs
        self.lr = lr
        self.params={}
        self.grads={}
        self.initialize_parameters()
        
    def sigmoid(self,Z):
        A = 1/(1+np.exp(-Z))
        return A

    def relu(self,Z):
        A = np.maximum(0,Z)
        return A

    def relu_backward(self,dA, cache):
        Z = cache
        dZ = np.array(dA)
        dZ[Z <= 0] = 0
        return dZ

    def sigmoid_backward(self,dA, cache):
        Z = cache 
        s = 1/(1+np.exp(-Z))
        dZ = dA * s * (1-s)
        return dZ
    
    def initialize_parameters(self):
        dims = self.dims
        params = self.params
        L = len(dims)
        for l in range(1,L):
            params["W"+str(l)] = np.random.randn(dims[l],dims[l-1]) * np.sqrt(1. /dims[l])
            params["b"+str(l)] = np.zeros((dims[l],1))
        self.params = params
    
    def forward_prop(self,X):
        L = len(self.dims)
        params = self.params
        activations = self.activations
        params['A0'] = X
        
        for l in range(1,L):
            params["Z"+str(l)] = np.dot(params["W"+str(l)], params["A"+str(l-1)]) + params["b"+str(l)]
            if activations[l-1] == "relu":    
                params["A"+str(l)] = self.relu(params["Z"+str(l)])
            elif activations[l-1] == "sigmoid":    
                params["A"+str(l)] = self.sigmoid(params["Z"+str(l)])
        self.params = params
    
    def compute_cost(self,Y):
        L = len(self.dims)
        params = self.params
        AL = params["A" + str(L-1)]
        m = Y.shape[1]
        mse = np.sum((np.power(AL-Y,2)),axis=0,keepdims=True)
        cost = (1/10) * np.sum((1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T)))
        return cost
    
    def back_prop(self,Y):
        L = len(self.dims)
        m = Y.shape[1]
        params = self.params
        grads = self.grads
        activations = self.activations
        AL = params["A" + str(L-1)] 
        dA = np.divide(AL - Y, (1 - AL) *  AL)
        
        for l in reversed(range(1,L)):
            if(activations[l-1] == "relu"):
                dZ = self.relu_backward(dA, params["Z"+str(l)])
            elif(activations[l-1] == "sigmoid"):
                dZ = self.sigmoid_backward(dA, params["Z"+str(l)])
            grads["dW"+str(l)] = (1/m) * np.dot(dZ, params["A"+str(l-1)].T)
            grads["db"+str(l)] = (1/m) * np.sum(dZ, axis=1, keepdims=True)
            dA = np.dot(params["W"+str(l)].T, dZ)
        self.params = params
        self.grads = grads
    
    def update_params(self):
        L = len(self.dims)
        lr = self.lr
        grads = self.grads
        params = self.params
        for l in range(1,L):
            params["W"+str(l)] -= grads["dW"+str(l)]*lr
            params["b"+str(l)] -= grads["db"+str(l)]*lr
        params = self.params
    
    def accuracy(self, X_val, Y_val):
        self.forward_prop(X_val)
        L = len(self.dims)
        Y_predict = self.params["A"+str(L-1)]
        Y_pred = np.argmax(Y_predict, axis = 0).reshape(1,-1)
        m = Y_val.shape[1]
        preds = ((Y_val) == (Y_pred))
        unique, counts = np.unique(preds, return_counts=True)
        preds = dict(zip(unique, counts))
        print(preds)
        accu = ((preds[True])/m)*100
        return accu
    
    def train(self, X, Y, X_val, Y_val):
        costs = []
        accus = []
        start_time = time.time()
        for i in range(self.epochs):
            self.forward_prop(X)
            
            cost = self.compute_cost(Y)
            self.back_prop(Y)
            
            self.update_params()
            
            accu = self.accuracy(X_val, Y_val)
            costs.append(cost)
            accus.append(accu)
            print('Epoch: {0}, Time Spent: {1:.2f}s, Accuracy: {2}, Cost: {3}'.format(
                i+1, time.time() - start_time, accu, cost))
        
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('epochs')
        plt.title("Learning rate =" + str(self.lr))
        plt.show()
        
        plt.plot(np.squeeze(accus))
        plt.ylabel('accuracy')
        plt.xlabel('epochs')
        plt.title("Learning rate =" + str(self.lr))
        plt.show()
            