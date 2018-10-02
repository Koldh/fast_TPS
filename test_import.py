from fast_TPS import *







mndata           = MNIST('/home/cosentino/python-mnist/data/')
X_train, Y_train = mndata.load_training()
Y_train          = asarray(Y_train.tolist())

#renormalizing
X_train         = asarray(X_train).reshape(shape(X_train)[0],1,28,28)

#getting indices for init
label0          = where(Y_train==0)[0]

X = asarray(X_train[label0[0]].reshape((28,28)).astype('float32'))
X /= X.max(axis=(0,1),keepdims=True)
X = X.reshape((1,1,28,28))


X_target = asarray(X_train[17].reshape((28,28)).astype('float32'))
X_target /= X_target.max(axis=(0,1),keepdims=True)
X_target = X_target.reshape((1,1,28,28))

lr = 0.003
num_c_points = 10
num_l_points = 12**2

x_shape = shape(X)


template = theano.shared(X)
y = T.ftensor4()

theta_control = theano.shared(zeros((num_c_points)).astype('float32'))

b = zeros((2,num_l_points+3), dtype='float32')
b[0,1]  = 1
b[1,2]  = 1
b = b.flatten()
layers       = lasagne.layers.DenseLayer((None,num_c_points),2*(num_l_points+3),b=b,nonlinearity=None)
params       = lasagne.layers.get_all_params(layers,trainable=True)
#get transform

l_in    = lasagne.layers.InputLayer((None,1,x_shape[2],x_shape[3]))
l_loc   = lasagne.layers.InputLayer((None,2*(num_l_points+3)))
l_trans = TPSTransformer(l_in,l_loc,control_points=num_l_points)

coefficients       = layers.get_output_for(theta_control).reshape((1,2,num_l_points+3))


template_transformed = l_trans.get_output_for([template,coefficients])


distance = template_transformed -  y
loss = (distance**2).mean()#+1.5*lasagne.regularization.l2(coefficients[:,:,3:])

update = lasagne.updates.adam(loss,[theta_control]+params,lr)
train  = theano.function([y],loss,updates=update)


get_template_trans =  theano.function([],template_transformed)
get_loss           = theano.function([y],loss)
get_grid_          = theano.function([],l_trans.get_grid())

err = []
for i in xrange(50):
        train(X_target.astype('float32'))
        err.append(get_loss(X_target.astype('float32')))
        print err[-1]


subplot(2,3,1)
imshow(X.reshape((28,28)),aspect='auto')
subplot(2,3,2)
imshow(get_template_trans().reshape((28,28)),aspect='auto')
subplot(2,3,3)
imshow(X_target.reshape((28,28)),aspect='auto')
subplot(2,3,5)
theta =  get_grid_()
print shape(theta)
source, dest = get_location(theta[0],num_l_points)
plot_grid(num_l_points,dest)
subplot(2,3,4)
plot_grid(num_l_points,source)

show()
