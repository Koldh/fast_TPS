from pylab import *
import theano
import theano.tensor as T
import lasagne
from lasagne.layers.base import Layer, MergeLayer
from lasagne.utils import as_tuple, floatX
from lasagne import nonlinearities
from lasagne import init
from scipy.io import loadmat
import matplotlib
from mnist import MNIST
from lasagne.regularization import regularize_layer_params_weighted, l2, l1
import sklearn.metrics
from sklearn.model_selection import train_test_split
import warnings
import scipy.ndimage
import time
from mpl_toolkits.mplot3d import Axes3D


class TPSTransformer(MergeLayer):
        def __init__(self, incoming, localization_network,downsample_factor=1,control_points=16,border_mode='nearest',**kwargs):
                super(TPSTransformer, self).__init__([incoming, localization_network], **kwargs)

                self.border_mode = border_mode
                self.downsample_factor = as_tuple(downsample_factor, 2)
                self.control_points = control_points

                input_shp, loc_shp = self.input_shapes

        #localization should output coefficient (batch_size,2,num_l_points+3)
        # Create source points and L matrix
                self.right_mat, self.source_points, self.out_height, self.out_width = _initialize_tps(control_points, input_shp, self.downsample_factor)


        def get_output_shape_for(self, input_shapes):
                shape = input_shapes[0]
                factors = self.downsample_factor
                return (shape[:2] + tuple(None if s is None else int(s // f)
                                  for s, f in zip(shape[2:], factors)))

        def get_output_for(self, inputs, **kwargs):
        # see eq. (1) and sec 3.1 in [1]
        # Get input and destination control points
                input, self.coefficients = inputs
                self.transformed_points, output =  _transform_thin_plate_spline(input, self.right_mat, self.coefficients,self.source_points, self.out_height, self.out_width, self.downsample_factor, self.border_mode)
                return output

        def get_grid(self):
                a= get_grid(self.control_points,self.coefficients)
                return a



def _transform_thin_plate_spline(input, right_mat, coefficients, source_points, out_height,out_width,downsample_factor, border_mode):

        num_batch, num_channels, height, width = input.shape
        num_control_points = source_points.shape[1]

        # reshape destination offsets to be (num_batch, 2, num_control_points)
        # and add to source_points

        # Transform each point on the source grid (image_size x image_size)
        right_mat = T.tile(right_mat.dimshuffle('x', 0, 1), (num_batch, 1, 1))
 
        transformed_points = T.batched_dot(coefficients, right_mat)
        #transformed_points_filtered  = T.tensordot(transformed_points,K,axes=([2],[1]))
        #transformed_points           = transformed_points_filtered.reshape((num_batch,2,784))



    # Get out new points
        x_transformed = transformed_points[:, 0].flatten()
        y_transformed = transformed_points[:, 1].flatten()

    # dimshuffle input to  (bs, height, width, channels)
        input_dim = input.dimshuffle(0, 2, 3, 1)
        input_transformed = _interpolate(input_dim, x_transformed, y_transformed,out_height, out_width, border_mode)

        output = T.reshape(input_transformed,(num_batch, out_height, out_width, num_channels))
        output = output.dimshuffle(0, 3, 1, 2)  # dimshuffle to conv format
        return transformed_points,output



def get_grid(num_control_points,coefficients):
        grid_size = int(sqrt(num_control_points))
        x_control_source, y_control_source = meshgrid(linspace(-1, 1, grid_size),linspace(-1, 1, grid_size))

    # Create 2 x num_points array of source points
        source_points = vstack((x_control_source.flatten(), y_control_source.flatten()))

    # Convert to floatX
        source_points = source_points.astype(theano.config.floatX)
        x_t, y_t = meshgrid(linspace(-1, 1, grid_size),linspace(-1, 1, grid_size))
        one = ones(prod(x_t.shape))
        orig_grid = vstack([x_t.flatten(), y_t.flatten(), one])
        orig_grid = orig_grid[0:2, :]
        orig_grid = orig_grid.astype(theano.config.floatX)

        to_transform = orig_grid[:, :, newaxis].transpose(2, 0, 1)
        stacked_transform = tile(to_transform, (num_control_points, 1, 1))
        stacked_source_points = source_points[:, :, newaxis].transpose(1, 0, 2)
        r_2 =  sum((stacked_transform - stacked_source_points) ** 2, axis=1)

    # Take the product (r^2 * log(r^2)), being careful to avoid NaNs
        log_r_2     = log(r_2)
        log_r_2[isinf(log_r_2)] = 0.
        distances   = r_2 * log_r_2

        # Add in the coefficients for the affine translation (1, x, and y,
        # corresponding to a_1, a_x, and a_y)
        num_batch = shape(coefficients)[0]
        upper_array        = ones(shape=(1, orig_grid.shape[1]),dtype=theano.config.floatX)
        upper_array        = concatenate([upper_array, orig_grid], axis=0)
        right_mat_land     = concatenate([upper_array, distances], axis=0)
        right_mat_land     = T.tile(right_mat_land[newaxis,:,:], (num_batch, 1, 1))
        transformed_points_land = T.batched_dot(coefficients, right_mat_land)
        return transformed_points_land


def _initialize_tps(num_control_points, input_shape, downsample_factor):

    # break out input_shape
        _, _, height, width = input_shape

    # Create source grid
        grid_size = sqrt(num_control_points)
        x_control_source, y_control_source = meshgrid(linspace(-1, 1, grid_size),linspace(-1, 1, grid_size))

    # Create 2 x num_points array of source points
        source_points = vstack((x_control_source.flatten(), y_control_source.flatten()))

    # Convert to floatX
        source_points = source_points.astype(theano.config.floatX)

    # Construct grid
        out_height = array(height // downsample_factor[0]).astype('int64')
        out_width = array(width // downsample_factor[1]).astype('int64')
        x_t, y_t = meshgrid(linspace(-1, 1, out_width),linspace(-1, 1, out_height))
        one = ones(prod(x_t.shape))
        orig_grid = vstack([x_t.flatten(), y_t.flatten(), one])
        orig_grid = orig_grid[0:2, :]
        orig_grid = orig_grid.astype(theano.config.floatX)

    # Construct right mat

        to_transform = orig_grid[:, :, newaxis].transpose(2, 0, 1)
        stacked_transform = tile(to_transform, (num_control_points, 1, 1))
        stacked_source_points = source_points[:, :, newaxis].transpose(1, 0, 2)
        r_2 =  sum((stacked_transform - stacked_source_points) ** 2, axis=1)

    # Take the product (r^2 * log(r^2)), being careful to avoid NaNs
        log_r_2     = log(r_2)
        log_r_2[isinf(log_r_2)] = 0.
        distances   = r_2 * log_r_2

        # Add in the coefficients for the affine translation (1, x, and y,
        # corresponding to a_1, a_x, and a_y)
        upper_array = ones(shape=(1, orig_grid.shape[1]),dtype=theano.config.floatX)
        upper_array = concatenate([upper_array, orig_grid], axis=0)
        right_mat   = concatenate([upper_array, distances], axis=0)
        # Convert to tensors
        out_height = T.as_tensor_variable(out_height)
        out_width = T.as_tensor_variable(out_width)
        right_mat = T.as_tensor_variable(right_mat)

        return right_mat, source_points, out_height, out_width




def _interpolate(im, x, y, out_height, out_width, border_mode):
            # *_f are floats
        num_batch, height, width, channels = im.shape
        height_f = T.cast(height, theano.config.floatX)
        width_f = T.cast(width, theano.config.floatX)

            # scale coordinates from [-1, 1] to [0, width/height - 1]
        x = (x + 1) / 2 * (width_f - 1)
        y = (y + 1) / 2 * (height_f - 1)

            # obtain indices of the 2x2 pixel neighborhood surrounding the coordinates;
            # we need those in floatX for interpolation and in int64 for indexing.
        x0_f = T.floor(x)
        y0_f = T.floor(y)
        x1_f = x0_f + 1
        y1_f = y0_f + 1

    # for indexing, we need to take care of the border mode for outside pixels.
        if border_mode == 'nearest':
                x0 = T.clip(x0_f, 0, width_f - 1)
                x1 = T.clip(x1_f, 0, width_f - 1)
                y0 = T.clip(y0_f, 0, height_f - 1)
                y1 = T.clip(y1_f, 0, height_f - 1)
        elif border_mode == 'mirror':
                w = 2 * (width_f - 1)
                x0 = T.minimum(x0_f % w, -x0_f % w)
                x1 = T.minimum(x1_f % w, -x1_f % w)
                h = 2 * (height_f - 1)
                y0 = T.minimum(y0_f % h, -y0_f % h)
                y1 = T.minimum(y1_f % h, -y1_f % h)
        elif border_mode == 'wrap':
                x0 = T.mod(x0_f, width_f)
                x1 = T.mod(x1_f, width_f)
                y0 = T.mod(y0_f, height_f)
                y1 = T.mod(y1_f, height_f)
        x0, x1, y0, y1 = (T.cast(v, 'int64') for v in (x0, x1, y0, y1))
            # The input is [num_batch, height, width, channels]. We do the lookup in
            # the flattened input, i.e [num_batch*height*width, channels]. We need
            # to offset all indices to match the flat version
        dim2 = width
        dim1 = width*height
        base = T.repeat(T.arange(num_batch, dtype='int64')*dim1, out_height*out_width)
        base_y0 = base + y0*dim2
        base_y1 = base + y1*dim2
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

            # use indices to lookup pixels for all samples
        im_flat = im.reshape((-1, channels))
        Ia = im_flat[idx_a]
        Ib = im_flat[idx_b]
        Ic = im_flat[idx_c]
        Id = im_flat[idx_d]

            # calculate interpolated values
        wa = ((x1_f-x) * (y1_f-y)).dimshuffle(0, 'x')
        wb = ((x1_f-x) * (y-y0_f)).dimshuffle(0, 'x')
        wc = ((x-x0_f) * (y1_f-y)).dimshuffle(0, 'x')
        wd = ((x-x0_f) * (y-y0_f)).dimshuffle(0, 'x')
        output = T.sum([wa*Ia, wb*Ib, wc*Ic, wd*Id], axis=0)
        return output


def get_location(theta_l,num_l_points):
        grid_size      = sqrt(num_l_points)
        theta_l        = theta_l.reshape((1,2,num_l_points))
        x_control_source, y_control_source = meshgrid(linspace(-1, 1, grid_size),linspace(-1, 1, grid_size))
        source_points  = vstack((x_control_source.flatten(), y_control_source.flatten()))
        dest_points    = theta_l
        dest_points_x  = (28)*(dest_points[0,0]+1)/2.
        dest_points_y  = (28)*(dest_points[0,1]+1)/2.
        source_points  = (28)*(source_points+1)/2.
        return source_points, [dest_points_x,dest_points_y]


def plot_grid(num_l_points,points):
        dim = num_l_points
        n_row_col = int(sqrt(dim))
        #get plot for horizontal lines
        p1_ind         = range(0,dim)
        to_be_removed = range(n_row_col-1,dim,n_row_col)
        for i in to_be_removed:
                p1_ind.remove(i)
        p1_ind      = asarray(p1_ind)
        p2_ind      = p1_ind +1
        plot(points[0][p1_ind[0]],points[1][p1_ind[0]],'ro')
        plot(points[0][n_row_col-1],points[1][n_row_col-1],'go')
        for i in xrange(len(p1_ind)):
                plot([points[0][p1_ind[i]],points[0][p2_ind[i]]],[points[1][p1_ind[i]],points[1][p2_ind[i]]],'y',linewidth=2.0)
        #get plot for vertical lines
        p1_ind      = range(0,dim-n_row_col)
        p2_ind      = range(n_row_col,dim)
        for i in xrange(len(p1_ind)):
                plot([points[0][p1_ind[i]],points[0][p2_ind[i]]],[points[1][p1_ind[i]],points[1][p2_ind[i]]],'y',linewidth=2.0)



