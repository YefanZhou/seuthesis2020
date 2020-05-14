'''
visualize in FoldingNetNew

author  : cfeng
created : 5/3/18 8:47 AM
'''
import sys
import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d, axis3d, proj3d
import argparse
import cv2
from sklearn.manifold import TSNE
import os
import tqdm
#import seaborn as sns
import math

def colormap2d(nx=45, ny=45):
    x = np.linspace(-3., 3., nx)
    y = np.linspace(-3., 3., ny)
    pos1, pos2 = np.meshgrid(x, y)
    X = pos1.ravel()
    Y = pos2.ravel()

    color2 = np.max([X+2.5,np.zeros(nx*ny)],axis=0)+np.max([Y+2.5,np.zeros(nx*ny)],axis=0)
    color3 = -np.min([Y-2.5,np.zeros(nx*ny)],axis=0)
    color1 = -np.min([X-2.5,np.zeros(nx*ny)],axis=0)

    color1 = (color1-color1.min())/(color1.max()-color1.min())
    color2 = (color2-color2.min())/(color2.max()-color2.min())
    color3 = (color3-color3.min())/(color3.max()-color3.min())
    clr=np.array([color1,color2,color3]).T
    return clr

def draw_pts(pts, clr, cmap, ax=None,sz=30):
    if ax is None:
        fig = plt.figure()
        ax = axes3d.Axes3D(fig)
        ax.view_init(-45,-64)
    else:
        ax.cla()
    pts -= np.mean(pts,axis=0) #demean

    ax.set_alpha(255)
#    ax.set_aspect('equal')
    min_lim = pts.min()
    max_lim = pts.max()
    ax.set_xlim3d(min_lim,max_lim)
    ax.set_ylim3d(min_lim,max_lim)
    ax.set_zlim3d(min_lim,max_lim)

    if cmap is None and clr is not None:
        #print(clr.shape)
        #print(pts.shape)
        assert(np.all(clr.shape==pts.shape))
        sct=ax.scatter(
            pts[:, 2], pts[:, 1], pts[:, 0],
            c=clr,
            zdir='y',
            s=sz,
            edgecolors=(0.5, 0.5, 0.5)
        )

    else:
        if clr is None:
            M = ax.get_proj()
            _,clr,_ = proj3d.proj_transform(pts[:,0], pts[:,1], pts[:,2], M)
        clr = (clr-clr.min())/(clr.max()-clr.min()) #normalization
        sct=ax.scatter(
            pts[:, 0], pts[:, 1], pts[:, 2],
            c=clr,
            zdir='y',
            s=sz,
            cmap=cmap,
            # depthshade=False,
            edgecolors=(0.5, 0.5, 0.5)
        )

    ax.set_axis_off()
    ax.set_facecolor("white")
    return ax,sct

def select_two_objects_for_interploation():
    ''' set args.aid and args.bid and return new args'''
    import pyxis
    db = pyxis.Reader(dirpath='data/shapenet57448xyzonly_16nn_GM_conv.lmdb')
    n_all_data = db.nb_samples

    nw = 3
    nh = 3
    n_obj = nw*nh
    try:
        sample_idx = np.random.permutation(n_all_data)[:n_obj]
        # sample_idx = np.array([20428,16, 22320, 25,42551, 15420, 2368,18963,45]) #51975, 11392
        while True:
            fig = plt.figure()

            all_ax = []
            for ith in xrange(n_obj):
                ax = fig.add_subplot(nw,nh,ith+1,projection='3d')
                all_ax.append(ax)

            selected = []
            def on_clicked(event):
                clicked = None
                for ith,ith_axes in enumerate(all_ax):
                    if ith_axes==event.inaxes:
                        clicked = ith
                        break
                if clicked is None:
                    return
                if clicked in selected:
                    selected.remove(clicked)
                    event.inaxes.patch.set_facecolor('white')
                else:
                    selected.append(clicked)
                    event.inaxes.patch.set_facecolor('yellow')
                event.canvas.draw()

            fig.canvas.mpl_connect('button_press_event',on_clicked)

            for ith,bid in enumerate(sample_idx):
                all_ax[ith].cla()
                sample = db.get_sample(bid)
                xyz = sample['data']
                draw_pts(xyz,None,'gray',ax=all_ax[ith])
            plt.suptitle('Select two objects to be interpolated:')
            plt.show()

            if len(selected)==2:
                # print(selected)
                # print(sample_idx)
                aid = sample_idx[selected[0]]
                bid = sample_idx[selected[1]]
                print('aid={}, bid={}'.format(aid, bid))
            else:
                print('Please select two and only two objects!')
                sample_idx = np.random.permutation(n_all_data)[:n_obj]
    except KeyboardInterrupt:
        print('User canceled!')
        exit(0)


def interactive_render_interploation(interp_file):
    from matplotlib.widgets import Slider
    fig = plt.figure(figsize=(12,4))
    ax_a = plt.subplot(131, projection='3d')
    ax_a.view_init(15,50)
    ax_i = plt.subplot(132, projection='3d')
    ax_i.view_init(15,50)
    ax_b = plt.subplot(133, projection='3d')
    ax_b.view_init(15,50)

    data=np.load(interp_file)
    draw_pts(data['Xa'],None,'gray',ax=ax_a)
    ax_a.set_title('Xa')
    draw_pts(data['Xb'],None,'gray',ax=ax_b)
    ax_b.set_title('Xb')

    clr2d = colormap2d()
    _,sct=draw_pts(data['all_Xp'][0],clr2d,'',ax=ax_i)
    ratio=0.01
    ax_i.set_title('Xa*({:.2f}) + Xb*(1-{:.2f})'.format(ratio, ratio))
    ax = plt.axes([0.1, 0.1, 0.8, 0.05])
    sr = Slider(ax, '', 0, 1, valinit=ratio)

    n_interp = len(data['all_Xp'])
    def update(val):
        ratio = sr.val
        ith = min(max(int(ratio*n_interp),0),n_interp-1)
        if ratio==0:
            draw_pts(data['Xpa'],clr2d,'plasma',ax=ax_i)
        elif ratio==1:
            draw_pts(data['Xpb'],clr2d,'plasma',ax=ax_i)
        else:
            draw_pts(data['all_Xp'][ith],clr2d,'plasma',ax=ax_i)
        ax_i.set_title('Xa*({:.2f}) + Xb*(1-{:.2f})'.format(ratio, ratio))
        fig.canvas.draw_idle()
    sr.on_changed(update)

    plt.subplots_adjust(left=0,right=1,bottom=0,top=1,wspace=0,hspace=0)
    plt.show()



def plot_embedding(data, colors, title, label):


    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

#    plt.scatter(data[:, 0], data[:, 1], c= colors ,label = label)
    plt.scatter(data[:, 0], data[:, 1], c= colors, s = 100, label = label)
    plt.xticks([])
    plt.yticks([])
    plt.legend(fontsize=30)
    plt.title(title, fontsize=30)
     

def tsne(codewords, colors, title, save_path, target_id, target_id_dic, cdscore):
    '''
    Input: 
        B x dim 
    '''
    c = ['b','c','g','k','m','r','y','#B8860B','#FF1493']
    

    codewords = codewords[target_id]
    print(target_id_dic.keys())
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    tsne_results = tsne.fit_transform(codewords)
    
    #fig = plt.figure(figsize=(30, 30))
    
    
    '''
    for key in tqdm.tqdm(target_id_dic, total= len(target_id_dic)):
        color = c[idx]
        plot_embedding(tsne_results[target_id_dic[key]], color, title, key)
        idx += 1
    '''
    idx = 0
    class_num = 0
    title = title +', chamfer distance = {}'.format(cdscore)
    for index, key in enumerate(target_id_dic):
        class_num += len(target_id_dic[key])
        plot_embedding(tsne_results[idx:class_num], c[index], title, label = key)
        print(len(target_id_dic[key]), idx, class_num, key)
        idx += len(target_id_dic[key])
        
    #fig.subplots_adjust(right=0.8)
    
    

def class_counter(args, split_name):
    f_class = open(os.path.join(args.data_basedir, args.splits_path, args.class_path),"r")
    
    class_num = 0
    class_dic = {}           # class_name : num of instance in this class
    class_index = {}         # class_name : class_index     e.g. airplane:0
    class_list = []          # 55 class
    data_class = []          # airairairchairchair
    color = []
    for line in f_class:
        index = line.find(' ')
        clname = line[:index]
        class_dic[clname] = 0
        class_list += [clname]
        class_index[clname] = class_num
        class_num += 1
        
    instance_num = 0
    for clname in tqdm.tqdm(class_list,total= len(class_list), desc = '%s img loading...'%split_name):
        f = open(os.path.join(args.data_basedir, args.splits_path, 'lists', clname, '%s.txt'%split_name),"r")
        for x in f:
            class_dic[clname] += 1
            instance_num += 1
            data_class += [clname]
            color += [class_index[clname]]
    
    #print(instance_num,class_dic, data_class) 
    return instance_num, class_dic, data_class, color


def load_pred(args, results_path, results_name, target_classnum = 10, total_num = 10432, origin_test_batch = 200):
    instance_num, class_dic, data_class, color = class_counter(args,split_name='test')

    sorted_class = sorted(class_dic.items(),key=lambda item:item[1],reverse = True)[:target_classnum]
    
    sorted_class = [item[0] for item in sorted_class]
    sorted_class.sort()
    print(sorted_class)

    target_id = []
    target_id_dic = {}
    for target_class in sorted_class:
        target_id_dic[target_class] = []

    for target_class in sorted_class:
        target_id += [i for i,x in enumerate(data_class) if x == target_class]
        target_id_dic[target_class] += [i for i,x in enumerate(data_class) if x == target_class]

    #print(target_id_dic)

    num_of_npy = math.ceil(total_num/origin_test_batch)
    
    for idx in tqdm.tqdm(range(num_of_npy)):
        postidx = '%03d'%idx + '.npy'
        tmp_pred = np.load(os.path.join(results_path, results_name + postidx))
        
        if idx == 0:
            pred = tmp_pred
        else:
            pred = np.concatenate((pred, tmp_pred), axis=0)
    #print(pred.shape)
    pred = np.squeeze(pred)
    print(pred.shape)
    color = np.array(color)
    color = np.expand_dims(color, axis=1)

    return pred, color, target_id, sorted_class,target_id_dic


def main(args): 
    '''
    exp_list = ["mixup_alpha1", 'foldingnet', "mixup_alpha0", "label5_ae"]
    logs_list = ['results_label5ae_0','results']
    log = logs_list[0]
    exp_list = [exp_list[3]]
    results_list = ['codeword_','fineptcloud_','img_','oriptcloud_','primiptcloud_',]
    
    results_name = results_list[0]

    fig, ax = plt.subplots(figsize=(30 * len(exp_list), 30))
    for idx, exp_name in enumerate(exp_list):
        plt.subplot(1, len(exp_list), idx+1)
        results_pth = '../experiment/{0}/{1}/final_vis'.format(exp_name, log)
        codewords, colors ,target_id, class_label, target_id_dic = load_pred(args, results_pth, results_name, target_classnum = 9, total_num = 10432, origin_test_batch = 200)
        #print(codewords.shape)
        tsne(codewords, colors, exp_name, args.save_path, target_id, target_id_dic, cdscore = 0.0716)   # label5_ae 1 0.0716


    plt.savefig(os.path.join(args.save_path,'tsne_label5ae_1.png'))
    '''

    


    for i in range(0,args.num,1):
        idx = str(args.num-1-i)
        print(int(idx))
        if len(idx) == 2:
            idx = '0'+ idx
        elif len(idx) == 1:
            idx = '00'+ idx
        print(os.path.join(args.results_path, 'img_{}.npy'.format(idx)))
        primptcloud = np.load(os.path.join(args.results_path, 'primiptcloud_{}.npy'.format(idx)))
        origin = np.load(os.path.join(args.results_path, 'oriptcloud_{}.npy'.format(idx)))
        img = np.load(os.path.join(args.results_path, 'img_{}.npy'.format(idx)))
        fineptcloud  = np.load(os.path.join(args.results_path, 'fineptcloud_{}.npy'.format(idx)))


        '''
        for t in range(fineptcloud.shape[0]):
    #        t = 8
            fig = plt.figure(figsize=(15,5))
            imge = np.transpose(img[t],(1,2,0))

            cv2.imshow('color',imge)
            clr = 0.3*np.ones((1024,3))
            ax_a = plt.subplot(131, projection='3d')
            ax_a.view_init(15,50)
            ax_b = plt.subplot(132, projection='3d')
            ax_b.view_init(15,50)
            ax_c = plt.subplot(133, projection='3d')
            ax_c.view_init(15,50)


            draw_pts(origin[t],clr,None,ax=ax_a)
            ax_a.set_title('GT_ptclpud')
    #        draw_pts(noise[0],clr,None,ax=ax_i)
    #        ax_i.set_title('cube')
            draw_pts(primptcloud[t],clr,None,ax=ax_b)
            ax_b.set_title('primptcloud')
            draw_pts(fineptcloud[t],clr,None,ax=ax_c)
            ax_c.set_title('fineptlcloud')
            plt.subplots_adjust(left=0,right=1,bottom=0,top=1,wspace=0,hspace=0)
            plt.show()
            print(t)
        
#            break 
#        break
        '''
        t = 16
    #    fineptcloud[t]
        imge = np.transpose(img[t],(1,2,0))
        fig = plt.figure(figsize=(10,10))
        cv2.imshow('color',imge)

        ax_c = plt.subplot(111, projection='3d')
        ax_c.view_init(-4,-5)
        clr = 0.2*np.ones((1024,3))
        #origin fineptcloud
        ptcloud = origin[t]

        draw_pts(ptcloud,clr,None,ax=ax_c)
#        draw_pts(origin[t],clr,None,ax=ax_c)
        plt.subplots_adjust(left=0,right=1,bottom=0,top=1,wspace=0,hspace=0)
#        plt.show()
        plt.savefig('../../figs/pro_state_demo/pro_state_origin.png')
        #ax_c.set_title('fineptlcloud')

if __name__ == '__main__':


    parser = argparse.ArgumentParser(sys.argv[0])
    parser.add_argument('--data-basedir',type=str,default='../../../What3D',
                        help='path of the jsonfile')
    parser.add_argument('--img-path',type=str,default='renderings',
                        help='path of the jsonfile')
    parser.add_argument('--splits-path',type=str,default='splits',
                        help='path of the jsonfile')
    parser.add_argument('--class-path',type=str,default='classes.txt',
                        help='path of the jsonfile')
#    parser.add_argument('--results-path',type=str,default='../experiment/label5_ae/results_label5ae_0/final_vis',
    parser.add_argument('--results-path',type=str,default='',
                        help='path of the jsonfile')
    parser.add_argument('--save-path',type=str,default='../img/mixup',
                        help='path of the jsonfile')
    parser.add_argument('--num',type=int,default=20,
                        help='number of result file')
    args = parser.parse_args(sys.argv[1:])
    
    print(str(args))
    main(args)
