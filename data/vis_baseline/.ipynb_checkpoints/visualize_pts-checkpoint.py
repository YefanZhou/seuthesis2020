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
import json 
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

def draw_pts(pts, clr, cmap, ax=None,sz=20):
    if ax is None:
        fig = plt.figure()
        ax = axes3d.Axes3D(fig)
        ax.view_init(-45,-64)
    else:
        ax.cla()
    pts -= np.mean(pts,axis=0) #demean

    ax.set_alpha(255)
    ax.set_aspect('equal')
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
            pts[:, 0], pts[:, 1], pts[:, 2],
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

    #ax.set_axis_off()
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

def main(args): 

    datalist = open(args.test_json,'r')
    datalist = json.load(datalist)
    interp_key = []
    for item in datalist:
        interp_key.append(round(100 * item["shapekey_value"]))

    interp_key.sort(reverse=False)
    '''
    for i in interp_key:
        i = str(int(i)).zfill(2)
        print(i)
        origin = np.load("%s/oriptcloud_interp_%s.npy" % (args.pt_dir, i))
        primptcloud = np.load("%s/primiptcloud_interp_%s.npy" % (args.pt_dir,i))
        fineptcloud = np.load("%s/fineptcloud_interp_%s.npy" % (args.pt_dir,i))
        
        img = np.load("%s/img_interp_%s.npy" % (args.pt_dir,i))
        print(fineptcloud)
        imge = np.transpose(img[0],(1,2,0))
        clr = colormap2d(32,32)
        plt.figure(figsize=(17, 10))
        ax_a = plt.subplot(142, projection='3d')
        ax_a.view_init(15,50)
        ax_b = plt.subplot(141, projection='3d')
        ax_b.view_init(15,50)
        ax_c = plt.subplot(143, projection='3d')
        ax_c.view_init(15,50)
        ax_d = plt.subplot(144)
        
        draw_pts(origin[0],clr,None,ax=ax_a)
        ax_a.set_title('GT_ptclpud_interpolation_{}%'.format(i))
        draw_pts(primptcloud[0],clr,None,ax=ax_b)
        ax_b.set_title('primptcloud')
        draw_pts(fineptcloud[0],clr,None,ax=ax_c)
        ax_c.set_title('fineptcloud_interpolation_{}%'.format(i))
        ax_d.imshow(imge)
        ax_d.set_title('interp_percentage_{}%'.format(i))
        plt.subplots_adjust(left=0,right=1,bottom=0,top=1,wspace=0,hspace=0)
        plt.show()
    '''
    
    for epoch in range(5, 305, 5):
        epoch = str(int(epoch)).zfill(3)
        interp = args.pt1_dir[-2:].zfill(3)
        interp_1 = args.pt2_dir[-2:].zfill(3)

        origin = np.load("%s/oriptcloud_epoch_%s_interp_%s.npy" % (args.pt1_dir, epoch, interp))
        primptcloud = np.load("%s/primiptcloud_epoch_%s_interp_%s.npy" % (args.pt1_dir, epoch, interp))
        fineptcloud = np.load("%s/fineptcloud_epoch_%s_interp_%s.npy" % (args.pt1_dir, epoch, interp))
        img = np.load("%s/img_epoch_%s_interp_%s.npy" % (args.pt1_dir, epoch, interp))

        origin_1 = np.load("%s/oriptcloud_epoch_%s_interp_%s.npy" % (args.pt2_dir, epoch, interp_1))
        primptcloud_1 = np.load("%s/primiptcloud_epoch_%s_interp_%s.npy" % (args.pt2_dir, epoch, interp_1))
        fineptcloud_1 = np.load("%s/fineptcloud_epoch_%s_interp_%s.npy" % (args.pt2_dir, epoch, interp_1))
        img_1 = np.load("%s/img_epoch_%s_interp_%s.npy" % (args.pt2_dir, epoch, interp_1))

        imge = np.transpose(img[0],(1,2,0))
        imge_1 = np.transpose(img_1[0],(1,2,0))

        clr = colormap2d(32,32)
        plt.figure(figsize=(17, 10))
        ax_a = plt.subplot(242, projection='3d')
        ax_a.view_init(15,50)
        ax_b = plt.subplot(241, projection='3d')
        ax_b.view_init(15,50)
        ax_c = plt.subplot(243, projection='3d')
        ax_c.view_init(15,50)
        ax_d = plt.subplot(244)

        #plt.figure(figsize=(17, 10))
        ax_e = plt.subplot(246, projection='3d')
        ax_e.view_init(15,50)
        ax_f = plt.subplot(245, projection='3d')
        ax_f.view_init(15,50)
        ax_g = plt.subplot(247, projection='3d')
        ax_g.view_init(15,50)
        ax_h = plt.subplot(248)
        
        draw_pts(origin[0],clr,None,ax=ax_a)
        ax_a.set_title('GT_ptclpud_epoch_{}_interp_{}%'.format(epoch, interp))
        draw_pts(primptcloud[0],clr,None,ax=ax_b)
        ax_b.set_title('primptcloud')
        draw_pts(fineptcloud[0],clr,None,ax=ax_c)
        ax_c.set_title('fineptcloud_epoch_{}_interp_{}%'.format(epoch, interp))
        ax_d.imshow(imge)
        ax_d.set_title('epoch_{}_interp_percentage_{}%'.format(epoch, interp))

        draw_pts(origin_1[0],clr,None,ax=ax_e)
        draw_pts(primptcloud_1[0],clr,None,ax=ax_f)
        draw_pts(fineptcloud_1[0],clr,None,ax=ax_g)
        ax_h.imshow(imge_1)

        plt.subplots_adjust(left=0,right=1,bottom=0,top=1,wspace=0,hspace=0)
        plt.show()
    



if __name__ == '__main__':


    parser = argparse.ArgumentParser(sys.argv[0])
    parser.add_argument('--test-json',type=str,default ='../../../../interp_data/test_interp.json',  #onehot_only 91 96 95 94   #GAN_only  GAN_onehot
                        help='test json file')
    parser.add_argument('--pt-dir',type=str,default ='final_vis',  #onehot_only 91 96 95 94   #GAN_only  GAN_onehot
                        help='first folder to be evaluate')
    parser.add_argument('--pt1-dir',type=str,default ='../results/interp_99',  #onehot_only 91 96 95 94   #GAN_only  GAN_onehot
                        help='second folder to be evaluate')
    parser.add_argument('--pt2-dir',type=str,default ='../results/interp_01',  #onehot_only 91 96 95 94   #GAN_only  GAN_onehot
                        help='second folder to be evaluate')
    parser.add_argument('-n', '--num', type=int,default = 300,
                        help='num of data')

    args = parser.parse_args(sys.argv[1:])
    
    print(str(args))
    main(args)
