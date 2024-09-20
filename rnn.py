import numpy as np

def categorical_cross_entropy_loss(y_pred, y_true):
    
    m = y_true.shape[0]
    
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    
    loss = -np.sum(y_true * np.log(y_pred)) / m
    
    return loss
    
def softmax(x, temperature = 1):

    x = x / temperature

    # Subtracting the max value for numerical stability
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class model:

    def __init__(self, input_size, output_size,temperature = 1, hidden_size=50, patience = 0.5 ) -> None:
        """input_size, output_size,temperature = 1, hidden_size=50"""
        self.trained = False
        self.y = None
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.parameters = {}
        self.loss = .0
        self.T = temperature
        self.patience = patience
        
    def initialize_parameters(self) -> None:
        
        input_size = self.input_size
        output_size = self.output_size
        hidden_size = self.hidden_size
    
        np.random.seed(0)
        
        w1 = np.random.randn(input_size, hidden_size ) * 0.01
        wh = np.random.randn(hidden_size, hidden_size) * 0.01
        w2 = np.random.randn( hidden_size, output_size) * 0.01
        
        b1 = np.random.randn(1,hidden_size) * 0.01 
        b2 = np.random.randn(1,output_size) * 0.01
        
        h = np.zeros((1,hidden_size)) # hidden state
        
        parameters = {"W1": w1,
                      "WH": wh,
                      "W2": w2,
                      "B1": b1,
                      "B2": b2,
                      "H": h}
        self.parameters = parameters
        
    def forward_propagation(self,x: np.array) -> None :
        
        parameters = self.parameters
        
        w1 = parameters["W1"]
        wh = parameters["WH"]
        w2 = parameters["W2"]
        b1 = parameters["B1"]
        b2 = parameters["B2"]
        h = parameters["H"]
        
        z1 = np.dot(x,w1)
        zh = np.dot(h,wh) + b1
        
        Ah = np.tanh( z1 + zh )
        
        y = np.dot(Ah,w2) + b2
        
        #activactionfunc is softmax for output layer
        y = softmax(y,self.T)
        parameters["H"] = Ah
        
        self.y = y
        self.parameters = parameters
          
        return y, parameters
          
    def backward_propagation(self,x, dy, learning_rate) -> dict:
    
        parameters = self.parameters
    
        w1 = parameters["W1"]
        wh = parameters["WH"]
        w2 = parameters["W2"]
        b1 = parameters["B1"]
        b2 = parameters["B2"]
        h = parameters["H"]
        
        dw2 = np.dot(h.T,dy)
        db2 = np.sum(dy, axis=0, keepdims=True)
        
        #tanh derivate
        tanh_derivate = ( 1 - h**2)
        
        dh = np.dot( dy , w2.T ) * tanh_derivate
        
        dwh = np.dot( h.T, dh)
        dw1 = np.dot( x.T, dh)
        db1 = np.sum(dh, axis=0, keepdims=True)
        
        #update parameters
        w1 -= learning_rate * dw1
        w2 -= learning_rate * dw2
        wh -= learning_rate * dwh
        b1 -= learning_rate * db1
        b2 -= learning_rate * db2
        
        parameters["W1"] = w1
        parameters["WH"] = wh
        parameters["W2"] = w2
        parameters["B1"] = b1
        parameters["B2"] = b2
        
        self.parameters = parameters
        
    def reset_state(self):
        self.parameters["H"] = np.zeros_like(self.parameters["H"])
        
    def train(self, inputs, targets, trainlimit, learning_rate):
        """
        Train the RNN model.
        inputs: List of input indices
        targets: List of one-hot encoded target vectors
        """
        
        input_size = self.input_size
        output_size = self.output_size
        hidden_size = self.hidden_size
        
        self.initialize_parameters()
        
        for j in range(trainlimit):
            
            loss = 0
            predictions = []
            
            # Reset hidden state at the beginning of each training iteration
            self.reset_state()
            
            for i in range(len(inputs)):
                x = np.zeros((1, input_size))
                x[0][inputs[i]] = 1
                
                y_pred,_ = self.forward_propagation(x)
                
                predictions.append(y_pred)
                
                target = targets[i].reshape(1, -1)  # Reshape to match the output shape

                dy = y_pred - target
                
                self.backward_propagation(x, dy, learning_rate/input_size)
                
                loss += categorical_cross_entropy_loss(y_pred, target)
            
            # Average loss over all time steps
            loss /= len(inputs)
            self.loss = loss
            
            if j % 100 == 0:
                print(f"Epoch {j}, loss: {loss:.4f}")
            if loss < self.patience:
                print(f"Epoch {j}, loss: {loss:.4f}")
                break

        self.reset_state()
        