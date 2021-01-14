
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
#import tensorflow as tf

#use v1 of tensorflow
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import data_utils
import viz
import re
import cameras, cameras2
import json
import os
from predict_3dpose import create_model
import cv2
import imageio
import time
import logging
import glob

from posenet import estimate_pose
from tools.utils import draw_3Dimg, draw_2Dimg, videoInfo, resize_img, common, midpoint



'''
FLAGS = tf.app.flags.FLAGS

order = [15, 12, 25, 26, 27, 17, 18, 19, 1, 2, 3, 6, 7, 8]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def main(_):
    done = []

    enc_in = np.zeros((1, 64))
    enc_in[0] = [0 for i in range(64)]

    actions = data_utils.define_actions(FLAGS.action)

    SUBJECT_IDS = [1, 5, 6, 7, 8, 9, 11]
    rcams = cameras2.load_cameras(FLAGS.cameras_path, SUBJECT_IDS)
    train_set_2d, test_set_2d, data_mean_2d, data_std_2d, dim_to_ignore_2d, dim_to_use_2d = data_utils.read_2d_predictions(
        actions, FLAGS.data_dir)
    train_set_3d, test_set_3d, data_mean_3d, data_std_3d, dim_to_ignore_3d, dim_to_use_3d, train_root_positions, test_root_positions = data_utils.read_3d_data(
        actions, FLAGS.data_dir, FLAGS.camera_frame, rcams, FLAGS.predict_14)

    device_count = {"GPU": 0}
    png_lib = []
    with tf.Session(config=tf.ConfigProto(
            device_count=device_count,
            allow_soft_placement=True)) as sess:
        #plt.figure(3)
        batch_size = 128
        model = create_model(sess, actions, batch_size)
        while True:
            key = cv2.waitKey(1) & 0xFF
            #logger.info("start reading data")
            # check for other file types
            list_of_files = glob.iglob("{0}/*".format(openpose_output_dir))  # You may use iglob in Python3
            latest_file = ""
            try:
                latest_file = max(list_of_files, key=os.path.getctime)
            except ValueError:
                #empthy dir
                pass
            if not latest_file:
                continue
            try:
                _file = file_name = latest_file
                print (latest_file)
                if not os.path.isfile(_file): raise Exception("No file found!!, {0}".format(_file))
                data = json.load(open(_file))
                #take first person
                _data = data["people"][0]["pose_keypoints_2d"]
                xy = []
                if len(_data)>=53:
                    #openpose incl. confidence score
                    #ignore confidence score
                    for o in range(0,len(_data),3):
                        xy.append(_data[o])
                        xy.append(_data[o+1])
                else:
                    #tf-pose-estimation
                    xy = _data

                frame_indx = re.findall("(\d+)", file_name)
                frame = int(frame_indx[-1])
                logger.debug("found {0} for frame {1}".format(xy, str(frame)))

                #body_25 support, convert body_25 output format to coco
                if len(xy)>54:
                    _xy = xy[0:19*2]
                    for x in range(len(xy)):
                        #del jnt 8
                        if x==8*2:
                            del _xy[x]
                        if x==8*2+1:
                            del _xy[x]
                        #map jnt 9 to 8
                        if x==9*2:
                            _xy[16] = xy[x]
                            _xy[17] = xy[x+1]
                        #map jnt 10 to 9
                        if x==10*2:
                            _xy[18] = xy[x]
                            _xy[19] = xy[x+1]         
                        #map jnt 11 to 10
                        if x==11*2:
                            _xy[20] = xy[x]
                            _xy[21] = xy[x+1]
                        #map jnt 12 to 11
                        if x==12*2:
                            _xy[22] = xy[x]
                            _xy[23] = xy[x+1]
                        #map jnt 13 to 12
                        if x==13*2:
                            _xy[24] = xy[x]
                            _xy[25] = xy[x+1]         
                        #map jnt 14 to 13
                        if x==14*2:
                            _xy[26] = xy[x]
                            _xy[27] = xy[x+1]
                        #map jnt 15 to 14
                        if x==15*2:
                            _xy[28] = xy[x]
                            _xy[29] = xy[x+1]
                        #map jnt 16 to 15
                        if x==16*2:
                            _xy[30] = xy[x]
                            _xy[31] = xy[x+1]
                        #map jnt 17 to 16
                        if x==17*2:
                            _xy[32] = xy[x]
                            _xy[33] = xy[x+1]
                        #map jnt 18 to 17
                        if x==18*2:
                            _xy[34] = xy[x]
                            _xy[35] = xy[x+1]
                    #coco 
                    xy = _xy

                joints_array = np.zeros((1, 36))
                joints_array[0] = [0 for i in range(36)]
                for o in range(len(joints_array[0])):
                    #feed array with xy array
                    joints_array[0][o] = xy[o]


                _data = joints_array[0]

                print("_data is ", _data)
                #at this pt _data should be array of 36

                
                # mapping all body parts or 3d-pose-baseline format
                for i in range(len(order)):
                    for j in range(2):
                        # create encoder input
                        enc_in[0][order[i] * 2 + j] = _data[i * 2 + j]

                #at this pt enc_in should be array of 64
                print("enc_in is ", enc_in)

                for j in range(2):
                    # Hip
                    enc_in[0][0 * 2 + j] = (enc_in[0][1 * 2 + j] + enc_in[0][6 * 2 + j]) / 2
                    # Neck/Nose
                    enc_in[0][14 * 2 + j] = (enc_in[0][15 * 2 + j] + enc_in[0][12 * 2 + j]) / 2
                    # Thorax
                    enc_in[0][13 * 2 + j] = 2 * enc_in[0][12 * 2 + j] - enc_in[0][14 * 2 + j]

                # set spine
                spine_x = enc_in[0][24]
                spine_y = enc_in[0][25]

                #at this pt enc_in should be array of 64
                #print("enc_in is ", enc_in)

                enc_in = enc_in[:, dim_to_use_2d]
                mu = data_mean_2d[dim_to_use_2d]
                stddev = data_std_2d[dim_to_use_2d]
                enc_in = np.divide((enc_in - mu), stddev)

                dp = 1.0
                dec_out = np.zeros((1, 48))
                dec_out[0] = [0 for i in range(48)]
                _, _, poses3d = model.step(sess, enc_in, dec_out, dp, isTraining=False)
                all_poses_3d = []
                enc_in = data_utils.unNormalizeData(enc_in, data_mean_2d, data_std_2d, dim_to_ignore_2d)
                poses3d = data_utils.unNormalizeData(poses3d, data_mean_3d, data_std_3d, dim_to_ignore_3d)
                gs1 = gridspec.GridSpec(1, 1)
                gs1.update(wspace=-0.00, hspace=0.05)  # set the spacing between axes.
                plt.axis('off')
                all_poses_3d.append( poses3d )
                enc_in, poses3d = map( np.vstack, [enc_in, all_poses_3d] )
                subplot_idx, exidx = 1, 1
                _max = 0
                _min = 10000

                for i in range(poses3d.shape[0]):
                    for j in range(32):
                        tmp = poses3d[i][j * 3 + 2]
                        poses3d[i][j * 3 + 2] = poses3d[i][j * 3 + 1]
                        poses3d[i][j * 3 + 1] = tmp
                        if poses3d[i][j * 3 + 2] > _max:
                            _max = poses3d[i][j * 3 + 2]
                        if poses3d[i][j * 3 + 2] < _min:
                            _min = poses3d[i][j * 3 + 2]

                for i in range(poses3d.shape[0]):
                    for j in range(32):
                        poses3d[i][j * 3 + 2] = _max - poses3d[i][j * 3 + 2] + _min
                        poses3d[i][j * 3] += (spine_x - 630)
                        poses3d[i][j * 3 + 2] += (500 - spine_y)

                # Plot 3d predictions
                ax = plt.subplot(gs1[subplot_idx - 1], projection='3d')
                ax.view_init(18, -70)    
                logger.debug(np.min(poses3d))
                if np.min(poses3d) < -1000 and frame != 0:
                    poses3d = before_pose

                p3d = poses3d

                viz.show3Dpose(p3d, ax, lcolor="#9b59b6", rcolor="#2ecc71")
                before_pose = poses3d
                pngName = 'png/test_{0}.png'.format(str(frame))
                plt.savefig(pngName)

                #plt.show()
                img = cv2.imread(pngName,0)
                rect_cpy = img.copy()
                cv2.imshow('3d-pose-baseline', rect_cpy)
                done.append(file_name)
                if key == ord('q'):
                    break
            except Exception as e:
                print (e)

        sess.close()



if __name__ == "__main__":

    openpose_output_dir = FLAGS.pose_estimation_json
    
    level = {0:logging.ERROR,
             1:logging.WARNING,
             2:logging.INFO,
             3:logging.DEBUG}

    logger.setLevel(level[FLAGS.verbose])


    tf.app.run()'''















































































FLAGS = tf.app.flags.FLAGS

order = [15, 12, 25, 26, 27, 17, 18, 19, 1, 2, 3, 6, 7, 8]
order_pnet_to_openpose = [0, 1, 7, 9, 11, 6, 8, 10, 13, 15, 17, 12, 14, 16, 3, 2, 5, 4]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

framenum = 0


#entry point for the tf app
def main(_):
    global framenum

    #set done to empty array, it will hold the json files from openpose that we've already processed
    done = []

    #initialize input tensor to 1x64 array of zeroes [[0. 0. 0. ...]]
    #this is list of numpy vectors to feed as encoder inputs (32 2d coordinates)
    enc_in = np.zeros((1, 64))
    enc_in[0] = [0 for i in range(64)]

    
    #actions to run on, default is all
    actions = data_utils.define_actions(FLAGS.action)

    #the list of Human3.6m subjects to look at
    SUBJECT_IDS = [1, 5, 6, 7, 8, 9, 11]


    #load camera parameters from the h36m dataset
    rcams = cameras2.load_cameras(FLAGS.cameras_path, SUBJECT_IDS)

    #loads 2d data from precomputed Stacked Hourglass detections
    train_set_2d, test_set_2d, data_mean_2d, data_std_2d, dim_to_ignore_2d, dim_to_use_2d = data_utils.read_2d_predictions(actions, FLAGS.data_dir)

    #loads 3d poses, zero-centres and normalizes them
    train_set_3d, test_set_3d, data_mean_3d, data_std_3d, dim_to_ignore_3d, dim_to_use_3d, train_root_positions, test_root_positions = data_utils.read_3d_data(
        actions, 
        FLAGS.data_dir, 
        FLAGS.camera_frame, rcams, 
        FLAGS.predict_14)

    device_count = {"GPU": 0}

    png_lib = []

    #run a tensorflow inference session
    with tf.Session(config=tf.ConfigProto(device_count=device_count, allow_soft_placement=True)) as sess:
        #plt.figure(3)

        #load pre-trained model
        batch_size = 128
        model = create_model(sess, actions, batch_size)

        #infinitely show 3d pose visualization
        while True:
            #wait for key to be pressed
            key = cv2.waitKey(1) & 0xFF

            _, frame = cv2.VideoCapture(0).read() #ignore the other returned value

            #resize and rotate the incoming image frame
            frame, W, H = resize_img(frame)
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)


            #try:
            #take 2d keypoints of first person (assume only one person). File looks like it actually has 18 joints (includes CHEST)
            #_data = data["people"][0]["pose_keypoints_2d"]

            #run posenet inference on the frame
            joints_2d = estimate_pose(frame)

            #throw out confidence score and flatten
            _data = joints_2d[..., :2].flatten()

            #print("data from posenet is ", _data)

            #open pop-up and draw the keypoints found
            #img2D  = draw_2Dimg(frame, joints_2d, 1)

            #fake the thorax point by finding midpt between left and right shoulder
            lt_should_x = _data[10]
            lt_should_y = _data[11]
            rt_should_x = _data[12]
            rt_should_y = _data[13]


            thorax = midpoint(lt_should_x, lt_should_y, rt_should_x, rt_should_y)

            #print("testing thorax pt at ", thorax)

            #insert thorax into data where it should be, at index 1
            _data = np.insert(_data, 2, thorax[0])
            _data = np.insert(_data, 3, thorax[1])

            #print("new _data is ", _data)
            _data = np.around(_data)


            #set xy to the array of 2d joint data
            xy = _data
            
            #create new 1x36 array of zeroes, which will store the 18 2d keypoints
            joints_array = np.zeros((1, 36))
            joints_array[0] = [0 for i in range(36)]

            #print("len of joints_array[0] is ", len(joints_array[0]))

            #iterates 18 times
            for o in range(int(len(joints_array[0]) / 2)):
                #feed array with xy array (the 18 keypoints), but switch ordering: posenet to openpose
                for j in range(2):
                    joints_array[0][o * (j+1) + j] = xy[order_pnet_to_openpose[o] * (j+1) + j]

            #set _data to the array containing the 36 coordinates of the 2d keypts
            _data = joints_array[0]


            #print("_data is ", _data)

            
            #mapping all body parts for 3d-pose-baseline format (32 2d coordinates)
            for i in range(len(order)): #iterates 14 times
                #select which coordinateof this point: x or y
                for j in range(2):
                    #create encoder input, switching around the order of the joint points
                    enc_in[0][order[i] * 2 + j] = _data[i * 2 + j]

            #now enc_in contains 14 points (28 total coordinates)

            #at this pt enc_in should be array of 64 vals

            for j in range(2):
                #place hip at index 0
                enc_in[0][0 * 2 + j] = (enc_in[0][1 * 2 + j] + enc_in[0][6 * 2 + j]) / 2
                #place neck/nose at index 14
                enc_in[0][14 * 2 + j] = (enc_in[0][15 * 2 + j] + enc_in[0][12 * 2 + j]) / 2
                #place thorax at index 13
                enc_in[0][13 * 2 + j] = 2 * enc_in[0][12 * 2 + j] - enc_in[0][14 * 2 + j]

            #set spine found by openpose
            spine_x = enc_in[0][24]
            spine_y = enc_in[0][25]

            #dim_to_use_2d is always [0  1  2  3  4  5  6  7 12 13 14 15 16 17 24 25 26 27 30 31 34 35 36 37 38 39 50 51 52 53 54 55]

            
            #take 32 entries of enc_in
            enc_in = enc_in[:, dim_to_use_2d]

            
            #find mean of 2d data
            mu = data_mean_2d[dim_to_use_2d]

            #find stdev of 2d data
            stddev = data_std_2d[dim_to_use_2d]

            #subtract mean and divide std for all
            enc_in = np.divide((enc_in - mu), stddev)

            #dropout keep probability
            dp = 1.0

            #output tensor, initialize it to zeroes. We'll get 16 joints with 3d coordinates
            #this is list of numpy vectors that are the expected decoder outputs
            dec_out = np.zeros((1, 48))
            dec_out[0] = [0 for i in range(48)]

            
            #get the 3d poses by running the 3d-pose-baseline inference. Model operates on 32 points
            _, _, poses3d = model.step(sess, enc_in, dec_out, dp, isTraining=False)
            #poses3d comes back as a 1x96 array (I guess its 32 points)

            #hold our 3d poses while we're doing some post-processing
            all_poses_3d = []

            #un-normalize the input and output data using the means and stdevs
            enc_in = data_utils.unNormalizeData(enc_in, data_mean_2d, data_std_2d, dim_to_ignore_2d)
            poses3d = data_utils.unNormalizeData(poses3d, data_mean_3d, data_std_3d, dim_to_ignore_3d)

            #create a grid for drawing
            gs1 = gridspec.GridSpec(1, 1)

            #set spacing between axes
            gs1.update(wspace=-0.00, hspace=0.05)  
            plt.axis('off')

            #fill all_poses_3d with the 3d poses predicted by the model step fxn
            all_poses_3d.append(poses3d)

            #vstack stacks arrays in sequence vertically (row wise)
            #this doesn't do anything in this case, as far as I can tell
            enc_in, poses3d = map(np.vstack, [enc_in, all_poses_3d])

            subplot_idx, exidx = 1, 1
            _max = 0
            _min = 10000

            #iterates once
            for i in range(poses3d.shape[0]):
                #iterate over all 32 points in poses3d
                for j in range(32):
                    #save the last coordinate of this point into tmp
                    tmp = poses3d[i][j * 3 + 2]

                    #swap the second and third coordinates of this pt
                    poses3d[i][j * 3 + 2] = poses3d[i][j * 3 + 1]
                    poses3d[i][j * 3 + 1] = tmp

                    #keep track of max of last coordinate
                    if poses3d[i][j * 3 + 2] > _max:
                        _max = poses3d[i][j * 3 + 2]
                    if poses3d[i][j * 3 + 2] < _min:
                        _min = poses3d[i][j * 3 + 2]

            #iterates once
            for i in range(poses3d.shape[0]):
                #iterate over all 32 points in poses3d (2nd and 3rd coords have all been swapped at this pt)
                for j in range(32):
                    #change the third coord of this pt, subtracting it from sum of max and min third coord to get new value
                    poses3d[i][j * 3 + 2] = _max - poses3d[i][j * 3 + 2] + _min

                    #modify first coord of this pt by adding the x coord of the spine found by 2d model
                    poses3d[i][j * 3] += (spine_x - 630)

                    #modify third coord of this pt by adding 500 minus y coord of spine found by 2d model
                    poses3d[i][j * 3 + 2] += (500 - spine_y)

            #Plot 3d predictions
            ax = plt.subplot(gs1[subplot_idx - 1], projection='3d')
            ax.view_init(18, -70)    
            logger.debug(np.min(poses3d))

            #if something happened with the data, reuse data from last frame
            if np.min(poses3d) < -1000 and frame != 0:
                poses3d = before_pose

            p3d = poses3d

            #plot the 3d skeleton
            viz.show3Dpose(p3d, ax, lcolor="#9b59b6", rcolor="#2ecc71")

    
            #keep track of this poses3d in case we need to reuse it for next frame
            before_pose = poses3d

            #save this frame as a png in the ./png/ folder
            pngName = 'png/test_{0}.png'.format(str(framenum))
            #print("pngName is ", pngName)

            
            plt.savefig(pngName)

            #plt.show()
            
            #read this frame which was just saved as png
            img = cv2.imread(pngName, 0)

            rect_cpy = img.copy()

            #show this frame
            cv2.imshow('3d-pose-baseline', rect_cpy)

            framenum+=1

            #quit if q is pressed
            if key == ord('q'):
                break


        sess.close()
        
        
        
if __name__ == "__main__":

    openpose_output_dir = FLAGS.pose_estimation_json
    
    level = {0:logging.ERROR,
             1:logging.WARNING,
             2:logging.INFO,
             3:logging.DEBUG}

    logger.setLevel(level[FLAGS.verbose])

    #run main() of our tensorflow app
    tf.app.run()
