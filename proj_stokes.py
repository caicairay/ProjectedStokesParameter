# Import Packages 
import numpy as np
import h5py
import yt
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

        self.LOS_generator(theta,phi,unit='rad',main_axis = None)
            Define LOS vector using theta and phi.

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
        """
        Main axis is a special coordinate axis. It must be one of the cube axses.
        Given LOS, the POS north is the projection of main axis, and the POS east is defined accordingly. 
        If LOS = main axis, POS north and east are other 2 cube axses.
        """
        if main_axis is not None:
            if main_axis not in [[1,0,0],[0,1,0],[0,0,1]]:
                sys.exit("Main axis must be one of x, y or z axis (e.g. [1,0,0])")
            else:
                self.main_axis = np.asarray(main_axis)
        return self.main_axis

    def set_LOS(self,n=None):
        """ 
        Define LOS vector
        Input:
            n: define LOS vector to be n. If n is None, return current LOS vector
        """
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

    def _LOS_bfield(self,vecx,vecy,vecz):
        """ Help function to project magnetic field """
        # Project vecx,y,z onto LOS vector
        los_b = self.los[0]*vecx \
              + self.los[1]*vecy \
              + self.los[2]*vecz 
        return los_b
    def add_LOS_bfield(self):
        """ add derived LOS b-field to YT dataset"""
        def get_LOS_b(field,data):
            vecx = data['magnetic_x']
            vecy = data['magnetic_y']
            vecz = data['magnetic_z']
            los_b = self._LOS_bfield(vecx,vecy,vecz)
            return los_b
        if (self.los_created):
            self.ds.add_field(("index","magnetic_field_LOS"), function=get_LOS_b,take_log=False,force_override=True,sampling_type="cell")
            print ("YT magnetic_field_LOS added.")
        else:
            sys.exit("Error: LOS vector is not defined")
        return True

    def stokes_POS_component(self,vecx,vecy,vecz):
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
            Q,U = self.stokes_parameter(vecx,vecy,vecz)
            return Q
        def get_U(field,data):
            vecx = data['magnetic_x']
            vecy = data['magnetic_y']
            vecz = data['magnetic_z']
            Q,U = self.stokes_parameter(vecx,vecy,vecz)
            return U

        if (self.los_created):
            self.ds.add_field(("index","stokes_Q"), function=get_Q,take_log=False,force_override=True,sampling_type="cell")
            self.ds.add_field(("index","stokes_U"), function=get_U,take_log=False,force_override=True,sampling_type="cell")
            print ("YT derived_field added.")
        else:
            sys.exit("Error: LOS vector is not defined")
        return True

    def create_window(self, center, width, resolution,north_vector):
        """ 
        Create projection window 
        Input:
            center: the center of the window
            width: the width of the window
            resolution: number of grids of the window
            north_vector: the vector that point 'up' after projection
        """
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
        """ 
        Project density on a plane 
        Input:
            data_object: a yt data object. Optional. Defalt is all_data
        """
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
        """
        Project stokes parameter on a plane 
        Input:
            data_object: a yt data object. Optional. Defalt is all_data
        """
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

    def project_LOS_b(self, data_object=None):
        """
        Integral LOS magnetic field along LOS
        Input:
            data_object: a yt data object. Optional. Defalt is all_data
        """
        if data_object is None:
            ad = self.ds
        else:
            ad = data_object
        if ((self.add_LOS_bfield()) and (self.window_created)):
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
            self.integral_LOS_b = yt.off_axis_projection(ad,c, L, W, N,
                                              "magnetic_field_LOS",
                                              #weight="density",
                                              north_vector=north_vector,
                                              no_ghost=False,
                                              num_threads = 2)
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
    def make_plot(self,save=False, figtitle=None, pcolormesh_kwargs={}, quiver_kwargs={}):
        x = self.proj_density/self.proj_density.max()
        y = self.DOP*100
        xtitle = r'$N/N_{max}$'
        ytitle = r'$P[\%]$'
        fig, axs = plt.subplots(2, 1,
                       figsize=(5,6),
                       gridspec_kw={
                           'height_ratios': [2.5, 1]})
    
        # the 2D plot
        ax = axs[0]
        # make axis locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size='5%', pad=0)
        cax_quiver = divider.append_axes("bottom", size='5%', pad=0)
        # make heatmap plot
        pcolormesh_kwargs['cmap'] = pcolormesh_kwargs.get('cmap', 'cividis')
        pc = ax.pcolormesh(self.x_corner,self.x_corner,x, **pcolormesh_kwargs)
        # make quiver plot
        default_quiver_kwargs = dict(cmap='RdBu_r', clim=(0,100),
                          headlength=0, pivot='middle',
                          scale=15.,headaxislength=0, headwidth=1)
        quiveropts = quiver_kwargs.copy()
        for key, value in default_quiver_kwargs.items():
            quiveropts[key] = quiveropts.get(key, value)
        every = quiveropts.pop('plot_every', None)
        mask1 = (slice(None,None,every))
        mask2 = (slice(None,None,every),slice(None,None,every))
        xc = self.x_center[mask1]
        yc = self.x_center[mask1]
        I = np.sin(self.offset)[mask2]
        J = np.cos(self.offset)[mask2]
        P = y[mask2]
        qv = ax.quiver(xc,yc,I.T,J.T,P,**quiveropts)
        # finalizing 2D plot
        plt.colorbar(pc, cax=cax, ax = ax)
        plt.colorbar(qv, cax=cax_quiver, ax = ax, orientation='horizontal')
        ax.set_aspect(1.)
        ax.tick_params(right= False,top= False,left= False, bottom= False,
                labelright=False,labeltop=False,labelleft=False,labelbottom=False)
        ax.set_title(xtitle)
    
        # the 1D plot
        ax = axs[1]
        ax.semilogy(x.flatten(),y.flatten(),'o',fillstyle='none')
        ax.set_xlabel(xtitle) 
        ax.set_ylabel(ytitle) 

        # finalizing the figure
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

    def LOS_generator(self, theta,phi,unit='rad',main_axis = None):
        """
        Helper function to generate LOS vector
        Input:
            theta, phi: angle in spherical coordinate
            unit: 'rad' or 'deg'
            main_axis: set the main axis of the spherical coordinate. Defalt is
                'z' axis
        Output:
            LOS vector: the vector represent the line of sight
        """
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
        self.set_LOS(vec)
        return vec
