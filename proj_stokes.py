#!/Users/dra/anaconda3/envs/py37_astropy/bin/python
# Import Packages 
import numpy as np
import h5py
import yt
import sys
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

class ProjectStokes():
    """
    To calculate projected stokes parameters
    Usage:
        self.create_window(center, width, resolution,north_vector)
        Define projection window parameter
            center: the center of the window
            width: the width of the window
            resolution: number of grids of the window
            north_vector: the vector that point 'up' after projection

        self.set_LOS(n=None)
        Define LOS vector
            if n is None, return current LOS vector
            else, define LOS vector to be n

        self.project_stokes(data_object=None)
        project stokes parameter on 2D plane, to mimic observation
            if data_object is None, project the whole domain
            else project provided data_object
    Note:
        before self.project_stokes is called, projection window and LOS vector must be created.
    """
    def __init__(self):
        """
        Initialize the class 
        """
        self.window_created = False
        self.los_created = False
        self.main_axis = np.asarray([0,0,1])
    def load_scorpio_uniform(self,flnm):
        """ 
        scorpio uniform data loader
        Input:
            flnm: the name of the file to be opened
        """
        with h5py.File(flnm,'r') as file:
            leftBdry = file['leftBdry'][()]
            rightBdry = file['rightBdry'][()]
            ashape = tuple(file['nMesh'][()].astype('int'))
            bbox = np.array([file['leftBdry'][()],file['rightBdry'][()]]).T
            nbuf = int(file['nbuf'][()])
            domain = tuple([slice(nbuf,-nbuf)]*3)
            time = file['t'][()]
            data = {}
            data['density']=file['den'][domain].T
            data['magnetic_x']=0.5*(file['bxl'][domain].T+file['bxr'][domain].T)
            data['magnetic_y']=0.5*(file['byl'][domain].T+file['byr'][domain].T)
            data['magnetic_z']=0.5*(file['bzl'][domain].T+file['bzr'][domain].T)
        self.ds = yt.load_uniform_grid(data, 
                                  ashape,
                                  bbox=bbox,
                                  sim_time = time,
                                  nprocs=64,
                                  unit_system="code"
                                  )
        return self.ds
    def set_main_axis(self, main_axis=None):
        if main_axis is not None:
            if main_axis not in [[1,0,0],[0,1,0],[0,0,1]]:
                sys.exit("Main axis must be one of x, y or z axis (e.g. [1,0,0])")
            else:
                self.main_axis = np.asarray(main_axis)
        return self.main_axis

    def set_LOS(self,n=None):
        """ Define LOS vector """
        if n is not None:
            self.los = n/np.linalg.norm(n)
            ## Setup New Coordinates
            if np.all(n == self.main_axis):
                tmp1 = np.asarray([[0,0,1],[0,1,0],[1,0,0]])
                tmp2 = [elem for elem in tmp1 if not np.all(elem == self.main_axis)]
                self.pos_north=np.array(tmp2[0])
                self.pos_east=np.array(tmp2[1])
            else:
                self.pos_north=self.main_axis-self.main_axis.T.dot(self.los)*self.los
                self.pos_north=self.pos_north/np.linalg.norm(self.pos_north)
                self.pos_east=np.cross(self.pos_north,self.los)
                self.pos_east=self.pos_east/np.linalg.norm(self.pos_east)
            self.los_created = True
            print ("New coordinate created\n  Main axis is {}\n  LOS is {}\n  POS north is {}\n  POS east is {}"
                   .format(self.main_axis,
                           self.los,
                           self.pos_north,
                           self.pos_east))
        return self.los

    def bfield_projection(self,vecx,vecy,vecz):
         """ Help function to project magnetic field """
         # Project vecx,y,z onto pos_north, pos_east
         u = np.stack((vecx,vecy,vecz),axis=3)

         vec_north = self.pos_north[0]*vecx \
                   + self.pos_north[1]*vecy \
                   + self.pos_north[2]*vecz
         vec_east = self.pos_east[0]*vecx \
                  + self.pos_east[1]*vecy \
                  + self.pos_east[2]*vecz

         # Get the Headless Orientation of 2d vectors
         def headless_orien(vec_north,vec_east,mask=None):
             t2=2*np.arctan(vec_east/vec_north)
             vec_2t_north=np.cos(t2)
             vec_2t_east=np.sin(t2)
             return vec_2t_north,vec_2t_east
         Q,U=headless_orien(vec_north,vec_east)
         return Q,U
    def derive_field(self):
        """ add derived stokes parameter field to YT dataset """
        def get_Q(field,data):
            vecx = data['magnetic_x']
            vecy = data['magnetic_y']
            vecz = data['magnetic_z']
            Q,U = self.bfield_projection(vecx,vecy,vecz)
            return Q
        def get_U(field,data):
            vecx = data['magnetic_x']
            vecy = data['magnetic_y']
            vecz = data['magnetic_z']
            Q,U = self.bfield_projection(vecx,vecy,vecz)
            return U

        if (self.los_created):
            self.ds.add_field(("index","stokes_Q"), function=get_Q,take_log=False,force_override=True,sampling_type="cell")
            self.ds.add_field(("index","stokes_U"), function=get_U,take_log=False,force_override=True,sampling_type="cell")
            print ("YT derived_field added.")
        else:
            sys.exit("Error: LOS vector is not defined")
        return True

    def create_window(self, center, width, resolution,north_vector):
        """ Create projection window """
        self.window_center = center
        self.window_width = width
        self.window_resolution = resolution
        if isinstance(north_vector, str):
            if north_vector != ("pos_north" or "pos_east"):
                sys.exit("Error: north_vector key word invalid")
            else:
                self.window_north_vector = north_vector
        else:
            self.window_north_vector = north_vector
        self.window_created = True
        print ("Window created.\n  window_center = {}\n  window_width = {}\n  window_resolution = {}\n  window_north_vector = {}"
                .format(np.array(self.window_center),
                        np.array(self.window_width),
                        np.array(self.window_resolution),
                        np.array(self.window_north_vector)))
        return self.window_resolution

    def project_density(self, data_object=None):
        """ Project density on a plane """
        if data_object is None:
            ad = self.ds
        else:
            ad = data_object
        if ((self.derive_field()) and (self.window_created)):
            c = self.window_center
            L = self.los
            W = self.window_width
            N = self.window_resolution
            if isinstance(self.window_north_vector, str):
                if self.window_north_vector == "pos_north":
                    north_vector = self.pos_north
                elif self.window_north_vector == "pos_east":
                    north_vector = self.pos_east
            else:
                north_vector = self.window_north_vector
            self.proj_density = yt.off_axis_projection(ad,c, L, W, N, "density",
                                              north_vector=north_vector,
                                              no_ghost=False,
                                              num_threads = 2)
        else:
            sys.exit("Error: Projection window is not defined")
        return True
    def project_stokes(self, data_object=None):
        """ Project stokes parameter on a plane """
        if data_object is None:
            ad = self.ds
        else:
            ad = data_object
        if ((self.derive_field()) and (self.window_created)):
            c = self.window_center
            L = self.los
            W = self.window_width
            N = self.window_resolution
            if isinstance(self.window_north_vector, str):
                if self.window_north_vector == "pos_north":
                    north_vector = self.pos_north
                elif self.window_north_vector == "pos_east":
                    north_vector = self.pos_east
            else:
                north_vector = self.window_north_vector
            self.proj_Q = yt.off_axis_projection(ad,c, L, W, N, "stokes_Q",
                                              weight="density",
                                              north_vector=north_vector,
                                              no_ghost=False,
                                              num_threads = 2)
            self.proj_U = yt.off_axis_projection(ad,c, L, W, N, "stokes_U",
                                              weight="density",
                                              north_vector=north_vector,
                                              no_ghost=False,
                                              num_threads = 2)
            self.DOP=np.sqrt(self.proj_Q**2+self.proj_U**2)
            self.offset=.5*np.arctan2(self.proj_U,self.proj_Q)
        else:
            sys.exit("Error: Projection window is not defined")
        return True

    def build_coords(self):
        """ 
        Build coordinates of according to given resolution
        help function for plotting.
        Output:
            self.x_corner : cell corner coordinate
            self.x_center : cell center coordinate
        Note:
            x_corner has larger dimensions than to be plotted field, see
            documentation of matplotlib.pyplot.pcolormesh
        """
        width = np.array(self.window_width)[0]
        resolution = self.window_resolution
        left_edge = -width/2
        right_edge = width/2
        dx = width/resolution
        self.x_corner = np.arange(left_edge, right_edge+dx/2, dx)
        self.x_center = self.x_corner[:-1]+dx/2
        return True
    def make_plot(self,save=False, figtitle=None):
        def annotation_polar(ax,x,y,Q,U,mask_every=None):
            vx = np.sin(self.offset)
            vy = np.cos(self.offset)
            quiveropts = dict(color='silver',
                              headlength=0, pivot='middle',
                              scale=15.,headaxislength=0, headwidth=1) 
            if mask_every is not None:
                dlength=len(Q.flatten())
                dshape=Q.T.shape
                mask = np.full(dlength, True)
                mask[:int(np.round(dlength/mask_every))] = False 
                np.random.shuffle(mask) 
                mask=np.reshape(mask,dshape)
                ma_vx=np.ma.array(vx,mask=mask)
                ma_vy=np.ma.array(vy,mask=mask)
                I = ma_vx
                J = ma_vy
            else:
                I = vx
                J = vy
            return ax.quiver(x,y,I.T,J.T,**quiveropts)
        def heatmap(ax,x,cmap,vmin=None,vmax=None):
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size='5%', pad=0)
            pc = ax.pcolormesh(self.x_corner,self.x_corner,x, cmap = cmap,vmin=vmin,vmax=vmax)
            qv = annotation_polar(ax,self.x_center,self.x_center,self.proj_Q,self.proj_U)
            plt.colorbar(pc, cax=cax, ax = ax)
            ax.set_aspect(1.)
            ax.tick_params(right= False,top= False,left= False, bottom= False,
                    labelright=False,labeltop=False,labelleft=False,labelbottom=False)
            return ax

        x = self.proj_density/self.proj_density.max()
        y = self.DOP*100
        xtitle = r'$N/N_{max}$'
        ytitle = r'$P[\%]$'
        xcmap = 'cividis'
        ycmap = 'RdBu_r'
        fig = plt.figure(figsize=(7,6))
    
        ax = plt.subplot(221)
        ax = heatmap(ax,x,xcmap,vmin=0,vmax=1)
        ax.set_title(xtitle)
    
        ax = plt.subplot(222)
        ax = heatmap(ax,y,ycmap,vmin=0,vmax=100)
        ax.set_title(ytitle)
    
        ax = plt.subplot(212)
        ax.loglog(x.flatten(),y.flatten(),'o')
        ax.set_xlabel(xtitle) 
        ax.set_ylabel(ytitle) 
        fig.tight_layout()
        if save:
            if figtitle is not None:
                fig.savefig(figtitle,dpi=300)
                print ("figure saved, name: {}".format(figtitle))
            else:
                sys.exit("Please define figtitle")
        else:
            plt.show()
    def dump_data(self,datatitle):
        x = self.proj_density.flatten()/self.proj_density.max()
        y = self.DOP.flatten()*100
        z = self.offset.flatten()
        array = np.asarray([x,y,z]).T
        np.savetxt(datatitle, array, delimiter=",")
        print ("Data dumpted, name: {}".format(datatitle))
    def los_generator(self, theta,phi,unit='rad',main_axis = None):
        if main_axis is None:
            main_axis = self.main_axis
        if unit == 'rad':
            pass
        elif unit == 'deg':
            theta = np.deg2rad(theta)
            phi = np.deg2rad(phi)
        else:
            sys.exit("Only 'rad' or 'deg' are acceptable")
        vx = np.sin(theta)*np.cos(phi)
        vy = np.sin(theta)*np.sin(phi)
        vz = np.cos(theta)
        if np.all(main_axis == [0,0,1]):
            vec = np.array([vx,vy,vz])
        elif np.all(main_axis == [1,0,0]):
            vec = np.array([vz,vx,vy])
        elif np.all(main_axis == [0,1,0]):
            vec = np.array([vy,vz,vx])
        else:
            sys.exit("main axis unacceptable")
        print ('LOS generated, LOS is {}'.format(vec))
        return vec



if __name__ == "__main__":
    flnm = '../data/perp/g0001_0023.h5'
    data = ProjectStokes()
    data.load_scorpio_uniform(flnm)

    center = data.ds.domain_center
    width = np.array([0.32]*3)
    resolution = 16
    data.create_window(center, width, resolution, "pos_north")
    data.build_coords()
    data.set_main_axis([1,0,0])
    theta = 90
    phi = 90
    n = data.los_generator(theta,phi,'deg',main_axis=[0,1,0])
    data.set_LOS(n)
#    ad = data.ds.sphere(center = data.ds.domain_center, radius = np.array(data.ds.domain_width)[0]/2)
    data.project_stokes()
    data.project_density()
    data.dump_data('test.txt')
    data.make_plot(save=True,figtitle='test.png')

#    res_case = 'low'
#    for obj in ['box','sphere']:
#        for win_size in ['small','large']:
#            for field in ['stokes','density']:
#                flnm = '../data/para/g0001_0023.h5'
#                data = ProjectStokes(flnm)
#                center = data.ds.domain_center
#                if win_size == 'small':
#                    width = np.array([0.32]*3)
#                    resolution = 16
#                elif win_size == 'large':
#                    width = data.ds.domain_width
#                    resolution = 18
#                data.create_window(center, width, resolution, "pos_north")
#                data.build_coords()
#            
#                np.random.seed(16062020)
#                los_list = np.random.rand(4,3)
#            
#                fig = plt.figure(figsize=(5,9))
#                grid = ImageGrid(fig, 111,
#                                 nrows_ncols=(1, 4),
#                                 axes_pad=0.0,
#                                 share_all=True,
#                                 label_mode="L",
#                                 cbar_location="right",
#                                 cbar_mode="single",
#                                 )
#                for los,ax in zip(los_list,grid):
#                    data.set_LOS(los)
#                    if obj == 'sphere':
#                        data.project_stokes(data.ds.sphere(center = data.ds.domain_center, radius = np.array(data.ds.domain_width)[0]/2))
#                        data.project_density(data.ds.sphere(center = data.ds.domain_center, radius = np.array(data.ds.domain_width)[0]/2))
#                    elif obj == 'box':
#                        data.project_stokes()
#                        data.project_density()
#                    if field == 'stokes':
#                        pc = ax.pcolormesh(data.x_corner,data.x_corner,data.DOP, vmin=0,vmax=1, cmap = 'RdBu_r')
#                    elif field == 'density':
#                        pc = ax.pcolormesh(data.x_corner,data.x_corner,np.log10(data.proj_density), 
#                                vmin=-1,vmax=3,cmap = 'cividis')
#                    qv = annotation_polar(ax,data.x_center,data.x_center,data.proj_Q,data.proj_U)
#                grid.cbar_axes[0].colorbar(pc)
#                fig.savefig("{}_{}_{}.png".format(obj,field,win_size),bbox_inches='tight',dpi=200)
