"""
Created on Dec 29 2019,
By zhixiang
"""

import os
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from model.Net import  STRNN
from utils.data_loader import basic_hyperparams, update_config, \
                              data_train_valid_test, shuffle_data, \
                              get_batch_input_dict
from utils.util import mkdir_file, scaled_laplacian, cheb_poly_approx, evaluation, z_inverse

def trainer(args):
    model_name = args.model_name
    np.random.seed(2019) 

    # use specific gpu
    os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    # set session
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    session = tf.Session(config=tf_config)

    # load train, valid, test, and adj data
    train_data, valid_data, test_data, adj, std, mean = data_train_valid_test(dataset=args.dataset,
                                                                              batch_size=args.batch_size, 
                                                                              len_his=args.len_his,
                                                                              len_his2=args.len_his2,
                                                                              len_pre=args.len_pre,
                                                                              len_f=args.len_f,
                                                                              model_name=args.model_name,
                                                                              merge=args.merge
                                                                              )
    mean = mean[:, :, 0]
    std = std[:, :, 0]

    # default config
    config = basic_hyperparams()

    # update config
    config = update_config(args, config)
    config.n_links = train_data['inputs'].shape[2]

    num_train = train_data['inputs'].shape[0]
    num_valid = valid_data['inputs'].shape[0]
    num_test = test_data['inputs'].shape[0]
    print('train: %s, test: %s, valid: %s' % (num_train, num_test, num_valid))

    if num_train % config.batch_size == 0:
        epoch_step = int(num_train // config.batch_size)
    else:
        epoch_step = int(num_train // config.batch_size) + 1
    config.decay_steps = args.decay_epoch * epoch_step

    # parameters
    valid_losses = [np.inf]

    # Calculate graph kernel
    L = scaled_laplacian(adj)
    # Alternative approximation method: 1st approx - first_approx(W, n).
    Lk = cheb_poly_approx(L, config.Ks, config.n_links)

    # model
    Net = STRNN
    with tf.name_scope('Train'):
        with tf.variable_scope('%s' % args.model_name, reuse=False):
            train_model = Net(config, Lk)
    with tf.name_scope('Valid'):
        with tf.variable_scope('%s' % args.model_name, reuse=True):
            valid_model = Net(config, Lk, is_training=False)


    logdir = args.base_path + \
    '%s/output/%s/%s_grp%s_mg%s_fcl%s_ep%s_bt%s_ly%s_u%s-%s_ks%s_kt%s_lr%s_dcr%s_dcp%s_dstp%s_dr%.1f_rg%.4f_%s_sp%s_%s-%s-%s_f-%s/' % \
    (model_name, model_name, config.dataset, args.isgraph, config.merge, config.isfcl, config.epoch, config.batch_size,\
    config.num_cell, config.filters[0], config.filters[1], config.Ks, config.Kt, config.lr, config.decay_rate, args.decay_epoch, config.global_step, \
    config.dr, config.rg, config.act_func, config.sampling, config.len_his, config.len_his2, config.len_pre, config.len_f)

    mkdir_file(logdir)
    model_dir = logdir
    print(logdir)

    mkdir_file('../output/')
    fp = open('../output/' + 'overall_metric_res.txt', 'a+')

    cnt = 0
    batch_size = config.batch_size
    # display_iter = 300
    save_log_iter = 200
    run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)
    lr = args.lr 
    with tf.Session() as sess:
        saver = tf.train.Saver()                        
        # initialize
        train_model.init(sess)
        iter = 0
        summary_writer = tf.summary.FileWriter(logdir)     

        for i in range(config.epoch):
            shuffle_train_data = shuffle_data(train_data)
            for l in range(0, num_train, batch_size):
                batch_train_data = {}
                iter+=1

                for key in shuffle_train_data.keys():
                    end_idx = l + batch_size
                    if end_idx > num_train-1:
                        end_idx = num_train - 1
                    batch_train_data[key] =  shuffle_train_data[key][l: end_idx]

                train_model.config.batch_size = batch_train_data['inputs'].shape[0]
                # print('==============================================preparing training data')
                input_feed_dict = get_batch_input_dict(train_model, batch_train_data, args.len_f)
                _, merged_summary = sess.run([train_model.phs['train_op'], train_model.phs['summary']], \
                                             feed_dict=input_feed_dict, \
                                             options=run_options)

            ########################## valid  start ##########################################################
            valid_loss = 0
            valid_preds = []
            for l in range(0, num_valid, batch_size):
                batch_valid_data = {}
                tp_idx = l + batch_size
                end_idx = min(tp_idx, num_valid)

                for key in valid_data.keys():
                    batch_valid_data[key] =  valid_data[key][l: end_idx]
                    valid_model.config.batch_size = batch_valid_data[key].shape[0]
                valid_input_feed_dict = get_batch_input_dict(valid_model, batch_valid_data, args.len_f)

                tmp_loss, tmp_pred = sess.run([valid_model.phs['loss'], valid_model.phs['preds']], \
                                    feed_dict=valid_input_feed_dict)
                valid_loss+=tmp_loss
                valid_preds.append(tmp_pred)
                # print('tmp_pred shape:\t', tmp_pred.shape)

            valid_loss /=  int(num_valid // batch_size)
            valid_losses.append(valid_loss)
            ########################## valid  end ##########################################################


            ########################## check valid loss save new model #########################################
            # save this model at after this epoch
            if valid_loss < min(valid_losses[:-1]):
                ## test
                cnt = 0
                valid_preds = np.concatenate(valid_preds, axis=0)
                if args.model_name == 'MSGcnLSTM':
                    valid_labels = valid_data['labels'][:,:,:,:,:2]
                else:
                    valid_labels = valid_data['labels']
                # print('-------------------------- valid labels shape:\t', valid_data['labels'].shape)
                # print('valid_preds shape:\t', valid_preds.shape)
                # print('valid_labels shape:\t', valid_labels.shape)

                # calculate metrics
                for horizon in range(1,config.len_pre+1, 1):
                    hr_preds = valid_preds[:, horizon-1:horizon,:].reshape(-1, config.n_links)
                    hr_labels = valid_labels[:, horizon-1:horizon,:].reshape(-1, config.n_links)
                    hr_preds = z_inverse(hr_preds, mean, std)
                    hr_labels = z_inverse(hr_labels, mean, std)
                    metrics_output = evaluation(hr_labels, hr_preds)
                    print(metrics_output)

                print('{} {} isgraph-{} epoch {} iter {}\tvalid_loss = {:.6f}\tmodel saved!!'.format(args.model_name, config.dataset, args.isgraph, i, iter, valid_loss)) 
                saver.save(sess, model_dir + 'final_model.ckpt')
                print(logdir)
            else:
                cnt += 1
                print('{} {} isgraph-{} epoch {} iter {}\tvalid_loss = {:.6f}\t'.format(args.model_name, config.dataset, args.isgraph, i, iter, valid_loss))  


            ########################## test ##########################################################
            if i % 10 == 9 or  cnt > 20:
                OVER_METRIC_DICT = {}
                preds = []
                for l in range(0, num_test, config.batch_size):
                    batch_test_data = {}
                    tp_idx = l + config.batch_size
                    end_idx = min(tp_idx, num_test)

                    for key in test_data.keys():
                        batch_test_data[key] =  test_data[key][l: end_idx]
                        valid_model.config.batch_size = batch_test_data[key].shape[0]

                    test_input_feed_dict = get_batch_input_dict(valid_model, batch_test_data, args.len_f)
                    batch_pred = sess.run(valid_model.phs['preds'], feed_dict=test_input_feed_dict)
                    # print('batch_pred shape:\t', batch_pred.shape)
                    preds.append(batch_pred)

                preds = np.concatenate(preds, axis=0)
                # print('preds shape:\t', preds.shape)
                labels = test_data['labels']
                print('valid_preds shape:\t', preds.shape) # (3557, 12, 170, 3, 1)
                print('valid_labels shape:\t', labels.shape)

                all_preds = []
                all_labels = []
                # calculate metrics
                for horizon in range(config.len_pre):
                    hr_preds = preds[:, horizon, :, :, 0] # [-1, n_links, len_his2]
                    hr_labels = labels[:, horizon, :, :, 0]
                    for ins in range(hr_preds.shape[-1]):
                        tp_pred = hr_preds[:, :,ins] #[-1, n_links]
                        tp_lable = hr_labels[:, :,ins]

                        tp_pred = z_inverse(tp_pred, mean, std)
                        tp_lable = z_inverse(tp_lable, mean, std)
                        metrics_output = evaluation(tp_lable, tp_pred)
                        print(metrics_output)
                        # print(tp_pred.shape)
                        all_preds.append(tp_pred) # [len_his2*len_pre, -1, n_links]
                        all_labels.append(tp_lable)
                        OVER_METRIC_DICT['%s-%s-%d' % (model_name, args.dataset, horizon*hr_preds.shape[-1]+ins+1)] = metrics_output

                print(np.asarray(all_preds).shape)
                all_preds = np.asarray(all_preds).transpose([1, 2, 0]) # [-1, n_links, len_his2*len_pre]
                all_labels = np.asarray(all_labels).transpose([1, 2, 0]) # [-1, n_links, len_his2*len_pre]

                saved_data_path = logdir + '%s/%s/' % (config.dataset, model_name) # /public/hezhix/DataParse/DurationPre/dnn/Pre/Baselines/saved_dat/
                saved_data_path = saved_data_path + '%s_grp%s_mg%s_fcl%s_ep%s_bt%s_ly%s_u%s_ks%s_kt%s_lr%s_dcr%s_dcp%s_dstp%s_dr%.1f_rg%.3f_%s_sp%s_%s-%s-%s_f-%s/' % \
                (config.dataset, args.isgraph, config.merge, config.isfcl, config.epoch,
                 config.batch_size, \
                 config.num_cell, config.filters[0], config.Ks, config.Kt, config.lr, config.decay_rate,
                 args.decay_epoch, config.global_step, \
                 config.dr, config.rg, config.act_func, config.sampling, config.len_his, config.len_his2,
                 config.len_pre, config.len_f)
                mkdir_file(saved_data_path)

                np.savez_compressed(saved_data_path + '%s-%s_%s_preds_gts_%s_%s-%s' % \
                                   (model_name, config.isgraph, config.dataset, config.act_func, \
                                   config.len_his2, config.len_pre),
                                   preds=np.float32(all_preds),
                                   gts=np.float32(all_labels))

                preds_12 = all_preds[:, :, :12]
                labels_12 = all_preds[:, :, :12]
                preds_12 = preds_12.transpose([0, 2, 1]).reshape(-1, config.n_links)
                labels_12 = labels_12.transpose([0, 2, 1]).reshape(-1, config.n_links)

                all_preds  = all_preds.transpose([0, 2, 1]).reshape(-1, config.n_links) # [-1, config.len_his2*config.len_pre, n_links]
                all_labels  = all_labels.transpose([0, 2, 1]).reshape(-1, config.n_links) # [-1, config.len_his2*config.len_pre, n_links]

                metric_output_12 = evaluation(labels_12, preds_12)
                metrics_output = evaluation(all_labels, all_preds)
                print('12:\t', metric_output_12)
                print('all:\t', metrics_output)


                fp.write('Epoch-%s_%s\t%s-%s\tmg%s\tfc%s\tLr%s-%s-Dc%s\tEp%s-Bt%s-Nc%s-Ut%s-Ks%s-Kt%s-Dr%s-sp%s\t%s-%s-%s' % (\
                        i, config.dataset, model_name, args.isgraph, args.merge, \
                        config.isfcl, config.lr, config.act_func, config.decay_rate,\
                        config.epoch, args.batch_size, config.num_cell,  \
                        config.filters[0], config.Ks, config.Kt, config.dr, \
                        config.sampling, config.len_his, config.len_his2, config.len_pre))



                for tm in list(metric_output_12)[:2]:
                    fp.write('\t%.2f' % tm)

                for tm in list(metrics_output)[:2]:
                    fp.write('\t%.2f' % tm)
                fp.write('\n')


                OVER_METRIC_DICT['%s-%s-%d-all' % (model_name, args.dataset, horizon+1)] = metrics_output
                OVER_METRIC_DICT['%s-%s-all-12' % (model_name, args.dataset)] = metric_output_12

                print('saving overall metrics results!!!')
                OVER_METRIC_df = pd.DataFrame.from_dict(OVER_METRIC_DICT, orient='index',)
                                                        # columns=['rmse', 'mae', 'mape', 'F_norm', 'r2', 'var'])
                over_metric_res_path = logdir + 'csv/' + 'overall_metric_res_%s.csv' % args.len_pre
                mkdir_file(over_metric_res_path)
                OVER_METRIC_df.to_csv(over_metric_res_path, index=True)
                # print(OVER_METRIC_df.values)
                # print(logdir)
            ########################## test end  #############################################################

            # early stop
            if valid_loss < 0.00001 or cnt > 20:
                print('training early stop!!!!!')
                break

            ########################## one epoch end  #############################################################

def tester(args):
    print('starting test======================== 1 ')
    model_name = args.model_name
    np.random.seed(2018)        
    
    # use specific gpu
    os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # set sesseion
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    session = tf.Session(config=tf_config)   
    print('starting test======================== 2')

    # load train, valid, test, and adj data
    train_data, valid_data, test_data, adj, std, mean = data_train_valid_test(dataset=args.dataset,
                                                                              batch_size=args.batch_size, 
                                                                              len_his=args.len_his,
                                                                              len_his2=args.len_his2,
                                                                              len_pre=args.len_pre,
                                                                              len_f=args.len_f,
                                                                              model_name=args.model_name,
                                                                              merge=args.merge
                                                                              )
    mean = mean[:, :, 0]
    std = std[:, :, 0]
    print('std, mean:\t', std.shape, mean.shape)
    # default config
    config = basic_hyperparams()

    # update config
    config = update_config(args, config)
    config.n_links = train_data['inputs'].shape[2]

    num_train = train_data['inputs'].shape[0]
    num_valid = valid_data['inputs'].shape[0]
    num_test = test_data['inputs'].shape[0]
    print('train: %s, test: %s, valid: %s' % (num_train, num_test, num_valid))

    if num_train % config.batch_size == 0:
        epoch_step = int(num_train // config.batch_size)
    else:
        epoch_step = int(num_train / config.batch_size) + 1

    # Calculate graph kernel
    L = scaled_laplacian(adj)
    # Alternative approximation method: 1st approx - first_approx(W, n).
    Lk = cheb_poly_approx(L, config.Ks, config.n_links)
    print('------------------------------------Lk shape:\t', Lk.shape)
    config.dr = 0

    # model
    Net = STRNN
    with tf.name_scope('Train'):
        with tf.variable_scope('%s' % args.model_name, reuse=False):
            train_model = Net(config, Lk)
    with tf.name_scope('Valid'):
        with tf.variable_scope('%s' % args.model_name, reuse=True):
            test_model = Net(config, Lk, is_training=False)


    logdir = args.base_path + \
    '%s/output/%s/%s_grp%s_mg%s_fcl%s_ep%s_bt%s_ly%s_u%s-%s_ks%s_kt%s_lr%s_dcr%s_dcp%s_dstp%s_dr%.1f_rg%.4f_%s_sp%s_%s-%s-%s_f-%s/' % \
    (model_name, model_name, config.dataset, args.isgraph, config.merge, config.isfcl, config.epoch, config.batch_size,\
    config.num_cell, config.filters[0], config.filters[1], config.Ks, config.Kt, config.lr, config.decay_rate, args.decay_epoch, config.global_step, \
    config.dr, config.rg, config.act_func, config.sampling, config.len_his, config.len_his2, config.len_pre, config.len_f)

    mkdir_file('../output/')
    fp = open('../output/' + 'test_overall_metric_res.txt', 'a+')

    # std = np.squeeze(std)
    # mean = np.squeeze(mean)

    model_dir = logdir
    if os.path.exists(model_dir+'checkpoint'):
        with tf.Session() as sess:

            saver = tf.train.Saver()    

            # initialize
            test_model.init(sess)
            saver.restore(sess, model_dir + 'final_model.ckpt')
            print('restore successfully')    
            test_loss = 0

            ## test
            OVER_METRIC_DICT = {}
            preds = []
            for l in range(0, num_test, config.batch_size):
                batch_test_data = {}
                tp_idx = l + config.batch_size
                end_idx = min(tp_idx, num_test)


                for key in valid_data.keys():
                    batch_test_data[key] =  test_data[key][l: end_idx]
                    test_model.config.batch_size = batch_test_data[key].shape[0]

                test_input_feed_dict = get_batch_input_dict(test_model, batch_test_data, args.len_f)
                batch_pred = sess.run(test_model.phs['preds'], feed_dict=test_input_feed_dict)
                # print('batch_pred shape:\t', batch_pred.shape)
                preds.append(batch_pred)

            preds = np.concatenate(preds, axis=0)
            print('preds shape:\t', preds.shape)

            if args.model_name == 'MSGcnLSTM':
                labels = test_data['labels'][:,:,:,:,:2]
            else:
                labels = test_data['labels']
            # print('-------------------------- valid labels shape:\t', valid_data['labels'].shape)
            # valid_preds = z_inverse(valid_preds, mean, std)
            # valid_labels = z_inverse(valid_labels, mean, std)
            print('test_preds shape:\t', preds.shape) # (3557, 12, 170, 3, 1)
            print('test_labels shape:\t', labels.shape)
            all_preds = []
            all_labels = []
            # calculate metrics
            for horizon in range(config.len_pre):
                hr_preds = preds[:, horizon, :, :, 0] # [-1, n_links, len_his2]
                hr_labels = labels[:, horizon, :, :, 0]
                for ins in range(hr_preds.shape[-1]):
                    tp_pred = hr_preds[:, :,ins] #[-1, n_links]
                    tp_lable = hr_labels[:, :,ins]

                    tp_pred = z_inverse(tp_pred, mean, std)
                    tp_lable = z_inverse(tp_lable, mean, std)
                    metrics_output = evaluation(tp_lable, tp_pred)
                    print(horizon*hr_preds.shape[-1]+ins+1, metrics_output)
                    # print(tp_pred.shape)
                    all_preds.append(tp_pred) # [len_his2*len_pre, -1, n_links]
                    all_labels.append(tp_lable)
                    OVER_METRIC_DICT['%s-%s-%d' % (model_name, args.dataset, horizon*hr_preds.shape[-1]+ins+1)] = metrics_output

            print(np.asarray(all_preds).shape)
            all_preds = np.asarray(all_preds).transpose([1, 2, 0]) # [-1, n_links, len_his2*len_pre]
            all_labels = np.asarray(all_labels).transpose([1, 2, 0]) # [-1, n_links, len_his2*len_pre]

            saved_data_path = logdir + '%s/test_3D/%ss/' % (config.dataset, model_name) # /public/hezhix/DataParse/DurationPre/dnn/Pre/Baselines/saved_dat/

            mkdir_file(saved_data_path)
            np.savez_compressed(saved_data_path + '%s-%s_%s_lr%s_ft%s-%spreds_gts_%s_%s-%s-%s' % \
                               (model_name, config.isgraph, config.dataset, config.lr, config.act_func, \
                                config.filters[0], config.filters[1], config.len_his, config.len_his2, \
                                config.len_pre), 
                               preds=np.float32(all_preds),
                               gts=np.float32(all_labels))

            all_preds  = all_preds.transpose([0, 2, 1]).reshape(-1, config.n_links) # [-1, config.len_his2*config.len_pre, n_links]
            all_labels  = all_labels.transpose([0, 2, 1]).reshape(-1, config.n_links) # [-1, config.len_his2*config.len_pre, n_links]

            metrics_output = evaluation(all_labels, all_preds)
            print('all:\t', metrics_output)


            fp.write('%s\t%s-%s\tmg%s\tfc%s\tLr%s-%s-Dc%s\tEp%s-Bt%s-Nc%s-Ut%s-%s-Ks%s-Kt%s-Dr%s-sp%s\t%s-%s-%s' % (\
                    config.dataset, model_name, args.isgraph, args.merge, \
                    config.isfcl, config.lr, config.act_func, config.decay_rate,\
                    config.epoch, args.batch_size, config.num_cell,  \
                    config.filters[0], config.filters[1],config.Ks, config.Kt, config.dr, \
                    config.sampling, config.len_his, config.len_his2, config.len_pre))

            for tm in list(metrics_output)[:2]:
                fp.write('\t%.2f' % tm)
            fp.write('\n')

            print(logdir)
    else:
        print('%s model not exists!!!' % logdir)
