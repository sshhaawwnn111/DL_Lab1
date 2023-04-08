import numpy as np
import matplotlib.pyplot as plt

def generate_linear(n=100):
    pts = np.random.uniform(0, 1, (n, 2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        distance = (pt[0]-pt[1])/1.414
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n, 1)

def generate_XOR_easy():
    inputs = []
    labels = []

    for i in range(11):
        inputs.append([0.1*i, 0.1*i])
        labels.append(0)

        if 0.1*i == 0.5:
            continue

        inputs.append([0.1*i, 1-0.1*i])
        labels.append(1)
    
    return np.array(inputs), np.array(labels).reshape(21, 1)

def show_result(x, y, pred_y, acc):
    plt.figtext(0, 0, "accuracy is " + str(acc), fontsize=14)
    plt.subplot(1,2,1)
    plt.title('Ground truth', fontsize=18)
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    plt.subplot(1,2,2)
    plt.title('Predict result', fontsize=18)
    for i in range(x.shape[0]):
        if pred_y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    plt.show()

class NeuralNet():
        
    def __init__(self, layers=[2, 13, 8, 1], learning_rate=0.001, iterations=100):
        self.params = {}
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.loss = []
        self.sample_size = None
        self.layers = layers
        self.X = None
        self.y = None
                
    def init_weights(self):
        np.random.seed(1) # Seed the random number generator
        self.params["W1"] = np.random.randn(self.layers[0], self.layers[1]) 
        self.params['b1'] = np.random.randn(self.layers[1],)
        self.params['W2'] = np.random.randn(self.layers[1],self.layers[2]) 
        self.params['b2'] = np.random.randn(self.layers[2],)
        self.params['W3'] = np.random.randn(self.layers[2],self.layers[3]) 
        self.params['b3'] = np.random.randn(self.layers[3],)
    
    def relu(self,Z):
        return np.maximum(0,Z)

    def dRelu(self, x):
        x[x<=0] = 0
        x[x>0] = 1
        return x

    def eta(self, x):
      ETA = 0.0000000001
      return np.maximum(x, ETA)

    def sigmoid(self,Z):
        return 1.0/(1.0 + np.exp(-Z))

    def dsigmoid(self, x):
        return np.multiply(x, 1.0 - x)

    def entropy_loss(self,y, yhat):
        nsample = len(y)
        yhat_inv = 1.0 - yhat
        y_inv = 1.0 - y
        yhat = self.eta(yhat) ## clips value to avoid NaNs in log
        yhat_inv = self.eta(yhat_inv) 
        loss = -1/nsample * (np.sum(np.multiply(np.log(yhat), y) + np.multiply((y_inv), np.log(yhat_inv))))
        return loss

    def MSE_loss(self,y, yhat):
        loss = np.mean(np.power(y - yhat, 2))
        return loss

    def dMSE(self, y, yhat):
        return -2 * np.mean(y - yhat)

    def forward_propagation(self):
        Z1 = self.X.dot(self.params['W1']) + self.params['b1']
        A1 = Z1
        Z2 = A1.dot(self.params['W2']) + self.params['b2']
        A2 = self.relu(Z2)
        Z3 = A2.dot(self.params['W3']) + self.params['b3']
        yhat = self.sigmoid(Z3)
        loss = self.entropy_loss(self.y, yhat)
            
        self.params['Z1'] = Z1
        self.params['Z2'] = Z2
        self.params['Z3'] = Z3
        self.params['A1'] = A1
        self.params['A2'] = A2

        return yhat,loss

    def back_propagation(self,yhat):
        dl_wrt_yhat = np.divide(1 - self.y, self.eta(1 - yhat)) - np.divide(self.y, self.eta(yhat))
        dl_wrt_sig = self.dsigmoid(yhat)
        dl_wrt_z3 = dl_wrt_yhat * dl_wrt_sig
        dl_wrt_b3 = dl_wrt_z3
        dl_wrt_w3 = self.params['A2'].T.dot(dl_wrt_z3)

        dl_wrt_A2 = dl_wrt_z3.dot(self.params['W3'].T)
        dl_wrt_z2 = dl_wrt_A2 * self.dRelu(self.params['Z2'])
        dl_wrt_b2 = dl_wrt_z2
        dl_wrt_w2 = self.params['A1'].T.dot(dl_wrt_z2)

        dl_wrt_A1 = dl_wrt_z2.dot(self.params['W2'].T)
        dl_wrt_z1 = dl_wrt_A1
        dl_wrt_b1 = dl_wrt_z1
        dl_wrt_w1 = self.X.T.dot(dl_wrt_z1)

        self.params['W1'] = self.params['W1'] - self.learning_rate * dl_wrt_w1
        self.params['W2'] = self.params['W2'] - self.learning_rate * dl_wrt_w2
        self.params['W3'] = self.params['W3'] - self.learning_rate * dl_wrt_w3
        self.params['b1'] = self.params['b1'] - self.learning_rate * dl_wrt_b1
        self.params['b2'] = self.params['b2'] - self.learning_rate * dl_wrt_b2
        self.params['b3'] = self.params['b3'] - self.learning_rate * dl_wrt_b3

    def fit(self, data_type):
        self.init_weights()

        for i in range(self.iterations):
            if data_type == "linear":
                Xtrain, ytrain = generate_linear()
            elif data_type == "xor":
                Xtrain, ytrain = generate_XOR_easy()
            # for j in range(len(Xtrain)):
            #     self.X = Xtrain[j].reshape(1, 2)
            #     self.y = ytrain[j].reshape(1, 1)
            #     yhat, loss = self.forward_propagation()
            #     self.back_propagation(yhat)

            self.X = Xtrain
            self.y = ytrain
            yhat, loss = self.forward_propagation()
            self.back_propagation(yhat)
            self.loss.append(loss)
            print("epoch", i, "loss :", loss)

    def predict(self, X):
        Z1 = X.dot(self.params['W1']) + self.params['b1']
        A1 = Z1
        Z2 = A1.dot(self.params['W2']) + self.params['b2']
        A2 = self.relu(Z2)
        Z3 = A2.dot(self.params['W3']) + self.params['b3']
        pred = self.sigmoid(Z3)
        return np.where(pred >= 0.5, 1, 0), pred

    def acc(self, y, yhat):
        acc = int(sum(y == yhat) / len(y) * 100)
        return acc

    def plot_loss(self):
        plt.plot(self.loss)
        plt.xlabel("Iteration")
        plt.ylabel("loss")
        plt.title("Loss curve for training")
        plt.show()

nn = NeuralNet(layers=[2, 4, 4, 1], learning_rate=0.001, iterations=700) # create the NN model
nn.fit("xor")
nn.plot_loss()

Xtest, ytest = generate_XOR_easy()
# Xtest, ytest = generate_linear()
test_pred, pred = nn.predict(Xtest)
print(pred.round(3))
print("Test accuracy is {}".format(nn.acc(ytest, test_pred)))

show_result(Xtest, ytest, test_pred, nn.acc(ytest, test_pred))

# fig, (ax1, ax2) = plt.subplots(1, 2)
# ax1.set_title('Ground truth')
# ax2.set_title('Predict result')
# ax1.scatter(Xtest[:, 0], Xtest[:, 1], c=ytest)
# ax2.scatter(Xtest[:, 0], Xtest[:, 1], c=test_pred)
# plt.show()
