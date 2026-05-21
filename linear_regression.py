import numpy as np
import matplotlib.pyplot as plt
#size of the house
x_train = np.array([0.8,1.0,1.2,1.5,1.8,2.0,2.2,2.5,2.8,3.0,3.2,3.5,3.8,4.0,4.5])
#cost of the house
y_train = np.array([220,300,360,430,500,560,610,680,760,820,890,980,1050,1120,1250])

#learning rate
alpha=0.01

#compute cost function
def compute_cost(x,y,w,b):
    #computation of the error function

    #number of training examples
    m=x.shape[0]

    #total cost variable
    total_cost=0

    for i in range(m):
        prediction=w*x[i]+b
        error=prediction-y[i]
        total_cost+=(error**2)

    #average cost
    total_cost=total_cost/(2*m)

    return total_cost    


#compute gradient

def compute_gradient(x,y,w,b):
    w_new = 0
    b_new = 0
    m=x.shape[0]
    for i in range(m):
        prediction=w*x[i]+b
        error=prediction-y[i]
        w_new=w_new+error*x[i]
        b_new=b_new+error

    w_new=w_new/m
    b_new=b_new/m   

    return w_new, b_new 

def gradient_descent(x,y,w_in,b_in,alpha,iterations):
    cost_history=[]
    w=w_in
    b=b_in
    for i in range (iterations):
    
        w_grad,b_grad=compute_gradient(x,y,w,b)
        w=w-alpha*w_grad
        b=b-alpha*b_grad

        cost=compute_cost(x,y,w,b)
        cost_history.append(cost)
        if(i%100==0):
            print("Interation:",i)
            print("Current cost:", cost)
            print("w:",w)
            print("b:",b)
    return w,b,cost_history

#initial values

w_init=0
b_init=0

alpha=0.01
iterations=1000

w_final,b_final,cost_history=gradient_descent(x_train,y_train,w_init,b_init,alpha,iterations)


print("Training completed:")
print("Final w:",w_final)
print("Final b:",b_final)


#accuracy of the algorithm
print("Accuracy of the algorithm:")
total_error_percent=0
for i in range(len(x_train)):
    prediction=w_final*x_train[i]+b_final
    error_precentage=abs((prediction-y_train[i])/y_train[i])*100
    total_error_percent=total_error_percent+error_precentage
avg_error_percent=total_error_percent/len(x_train)
print("Percentage of error:",avg_error_percent)
print("Accuracy:",100-avg_error_percent,"%") 

plt.scatter(x_train,y_train,label="Training Data")
x_line=np.linspace(0,5,100)
y_line=x_line*w_final+b_final
plt.plot(x_line,y_line,label="Regression Line")
plt.xlabel("House size")
plt.ylabel("Cost of the house")
plt.title("House price prediction")
plt.legend()
plt.grid(True)
plt.show()  

#making predictions
s=float(input("Enter the size of the house:"))
house_cost=w_final*s+b_final
print("Cost of the house:",house_cost)

 