class gnn:
    def __init__(self, dims, activations, epochs, lr, b_size):
        self.dims = dims
        self.activations = activations
        self.epochs = epochs
        self.lr = lr
        self.b_size = b_size
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
        #Change cost function accordingly
        L = len(self.dims)
        params = self.params
        AL = params["A" + str(L-1)]
        m = Y.shape[1]
        mse = np.sum((np.power(AL-Y,2)),axis=0,keepdims=True)
        cost = (1/10) * np.sum((1./m) * (np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T)))
        return cost
    
    def back_prop(self,Y):
        L = len(self.dims)
        m = Y.shape[1]
        params = self.params
        grads = self.grads
        activations = self.activations
        AL = params["A" + str(L-1)] 
        dA = np.divide(AL-Y, (1 - AL) *  AL)
        
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
        #Define how to get Y_pred based on your code implementation
        preds = ((Y_val) == (Y_pred))
        unique, counts = np.unique(preds, return_counts=True)
        preds = dict(zip(unique, counts))
        print(preds)
        accu = ((preds[True])/m)*100
        return accu
    
    def train(self, X, Y, X_val, Y_val):
        costs = []
        accus = []
        b_size = self.b_size
        m = Y.shape[1]
        n_batches = m//b_size
        start_time = time.time()
        for i in range(self.epochs):
            ll = 0
            for j in range(n_batches):
                ul = ll + b_size-1
                XT = X[:,ll:ul]
                YT = Y[:,ll:ul]
                self.forward_prop(XT)

                cost = self.compute_cost(YT)
                self.back_prop(YT)

                self.update_params()

                
                ll += b_size
                
            costs.append(cost)
            accu = self.accuracy(X_val, Y_val)
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
            
