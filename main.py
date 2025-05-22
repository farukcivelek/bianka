import cvfunc
import cv2
import os
import numpy as np
import time
import configparser


def main():
    # Init configparser to read ini-file
    config = configparser.ConfigParser()
    config.read('config.ini')
    
    # Define Rodirectories
    d1 = 'imgs/Durchgang1/'     # registration database
    d2 = 'imgs/Durchgang2/'     # identification database
    databases = [d1, d2]
    res = 'res/'                # folder to store processed iamges

    # RoI in original image (defined in config.ini)
    row1 = int(config['CV_PARAMS']['row1'])
    row2 = int(config['CV_PARAMS']['row2'])
    col1 = int(config['CV_PARAMS']['col1'])
    col2 = int(config['CV_PARAMS']['col2'])

    # First list index and last list index
    fst_index = int(config['EXEC_PARAMS']['fst_part'])
    if config['EXEC_PARAMS']['lst_part'] == 'ALL':
        lst_index = len(os.listdir(d1)) + 1
    else:
        lst_index = int(config['EXEC_PARAMS']['lst_part']) + 1

    # Start timer
    start = time.time()

    # Get CV parameters from config file
    thresh = int(config['CV_PARAMS']['thresh'])    #240 for 211012_FV815-0,5%_weiß_365nm
    nfeatures = int(config['CV_PARAMS']['nfeatures'])
    num_to_keep = int(config['CV_PARAMS']['num_to_keep']) #30 for 211012_FV815-0,5%_weiß_365nm
    min_rad = int(config['CV_PARAMS']['min_rad']) #50 for 211012_FV815-0,5%_weiß_365nm
    fpcol = (0,0,255)
    algor = config['CV_PARAMS']['algor']
    max_diff = config['CV_PARAMS']['max_diff']

    # Registration: pre-process images and find feature points
    if bool(int(config['EXEC_PARAMS']['register'])):
        for db in databases[0:]:
            for im_file in os.listdir(db)[fst_index:lst_index]:
                # Read image
                im = cv2.imread(db + im_file, 0)

                # Extract RoI and store image
                im = cvfunc.get_roi(im, row1, row2, col1, col2)
                cv2.imwrite(res + 'roi/' + db.split("/")[1] + '/' + im_file, im)

                # Binarize and store image
                im_bin = cvfunc.binar(im, thresh)
                cv2.imwrite(res + 'bin/' + db.split("/")[1] + '/' + im_file, im_bin)

                # Create text file to store keypoints' information
                tfile_dir = res + 'files/' + db.split("/")[1] + '/' + im_file.split(".")[0] + '.txt'
                tfile = open(tfile_dir, "w")

                # Find feature points and store image with illustrated feature points
                orb = cv2.ORB_create(nfeatures=nfeatures)
                kp, _ = orb.detectAndCompute(im_bin, None)
                im_fp = cv2.drawKeypoints(im, kp, None, color=fpcol, flags=0)        
                cv2.imwrite(res + 'featpts/' + db.split("/")[1] + '/' + im_file, im_fp)

                # Find subset of detected feature points having a uniform distribution
                kp_anms = cvfunc.adaptiveNonMaximalSupression2(kp, num_to_keep, min_rad)
                if len(kp_anms) < num_to_keep:
                    print('WARNING: adms algorithm returned less keypoints than expected')
                im_anms = cv2.drawKeypoints(im, kp_anms, None, color=fpcol, flags=0)
                cv2.imwrite(res + 'featpts-anms/' + db.split("/")[1] + '/' + im_file, im_anms)

                # Write keypoints to textfile
                for kp in kp_anms:
                    p = str(kp.pt[0]) + "," + str(kp.pt[1]) + "," + str(kp.size) + "," + \
                        str(kp.angle) + "," + str(kp.response) + "," + str(kp.octave) + "," + \
                        str(kp.class_id) + "\n"
                    tfile.write(p)
                
                # Close text file with  keypoints' information stored
                tfile.close()

                print('Descriptor file for ' + im_file + ' created')

    # Create log file and log important parameters
    logfile_dir = '{}/log_{}.txt'.format(res, algor)
    logfile = open(logfile_dir, "w")
    logfile.write('#### computer vision parameters ####\n')
    logfile.write('algorithm = {}\n\n'.format(algor))
    logfile.write('roi (row1,row2,col1,col2) = {},{},{},{}\n'.format(row1,row2,col1,col2))
    logfile.write('thresh = {}\n'.format(thresh))
    logfile.write('nfeatures = {}\n'.format(nfeatures))
    logfile.write('numtokeep = {}\n'.format(num_to_keep))
    logfile.write('minrad = {}\n\n'.format(min_rad))
    logfile.write('max_diff = {}\n\n'.format(max_diff))
    logfile.write('#### Identification results ####\n')

    # get time for register processes
    t1 = time.time()

    # variable to count successful identification processes
    success = 0

    # Compare images and find best match
    for file_d1 in os.listdir(d1)[fst_index:lst_index]:
        min_dist = 999999.9
        max_corrpts = 0

        # 1) read kps from file, 2) make kp list comprising coordinate-tuples only, 3) calculate image descriptor
        tfile_dir_d1 = res + 'files/' + d1.split("/")[1] + '/' + file_d1.split(".")[0] + '.txt'
        im_kp_d1 = cvfunc.read_kp_from_tfile(tfile_dir_d1)
        im_kp_d1_pt = [[round(item.pt[0], 1), round(item.pt[1], 1)] for item in im_kp_d1]
        desc_im_d1 = cvfunc.im_descr(im_kp_d1)
        
        for file_d2 in os.listdir(d2)[fst_index:lst_index]:
            tfile_dir_d2 = res + 'files/' + d2.split("/")[1] + '/' + file_d2.split(".")[0] + '.txt'
            im_kp_d2 = cvfunc.read_kp_from_tfile(tfile_dir_d2)
            im_kp_d2_pt = [[round(item.pt[0], 1), round(item.pt[1], 1)] for item in im_kp_d2]
            desc_im_d2 = cvfunc.im_descr(im_kp_d2)

            # perform identification dependent on selected algorithm (nn or dis)
            if algor == 'dis':
                corr_pts = 0

                # perform distance analysis
                corr_pts = cvfunc.kpdis_analysis(desc_im_d1, desc_im_d2)

                if corr_pts >= max_corrpts:
                    max_corrpts = corr_pts
                    best_match = file_d2

            elif algor == 'nn':
                # perform nearest neighbour analysis
                nn_ind, nn_kps, nn_dists, n, corr_pts = cvfunc.nn_analysis(im_kp_d1_pt, im_kp_d2_pt)
                
                if np.mean(nn_dists) < min_dist and corr_pts >= max_corrpts:
                    min_dist = np.mean(nn_dists)
                    max_corrpts = corr_pts
                    best_match = file_d2

            else: 
                'WARNING: algorihm not found. Please check config file.'

            # print(np.mean(nn_dists), corr_pts)

        print('{} has the best match with {}. Corresp points: {}'.format(file_d1, best_match, max_corrpts))
        logfile.write('{} has the best match with {}. Corresp points: {}\n'.format(file_d1, best_match, max_corrpts))
        # print('Corresponding points: ' + str(max_corrpts))
        # print('min_dist: ' + str(min_dist))
        
        if file_d1 == best_match:
            success += 1

    logfile.write('\n#### Total result ####\n')
    logfile.write('{} from {} parts successfully identified'.format(success, len(os.listdir(d2))))

    end = time.time()
    total_runt = end-start
    reg_runt = t1-start
    id_runt = end-t1
    print('Identification runtime: ' + str(id_runt) + ' sec')
    logfile.write('\n#### Runtime ####\n')
    logfile.write('Total runtime: {} sec\n'.format(total_runt))
    logfile.write('Total registration runtime: {} sec\n'.format(reg_runt))
    logfile.write('Registration runtime per part: {} sec\n'.format(reg_runt / (lst_index-fst_index)))
    logfile.write('Total identification runtime: {} sec\n'.format(id_runt))
    logfile.write('Identficiation runtime per part: {} sec\n'.format(id_runt / (lst_index-fst_index)))

    print('{} from {} parts successfully identified'.format(success, len(os.listdir(d2))))

    logfile.close()


if __name__ == '__main__':
    main()