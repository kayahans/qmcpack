#!/usr/bin/env python
from numpy import array,ndarray,dot,arange,atleast_2d,meshgrid,transpose,vstack,reshape,tile,round,pi,linspace, sin, cos, sqrt, where, zeros, log, exp, squeeze, inf, isclose, equal, sort
from numpy.linalg import inv, norm
from developer import DevBase,obj
from numerics import ndgrid
from structure import Structure
from scipy.ndimage.interpolation import map_coordinates
import pdb
import os
from fileio import XsfFile, ChgcarFile
from periodic_table import pt as ptable

# to be added to developer.py
class Missing:
    def __call__(self,value):
        return isinstance(value,Missing)
    #end def __call__
#end class Missing
missing = Missing()

def all_missing(*values):
    m = True
    for v in values:
        m &= isinstance(v,Missing)
    #end for
    return m
#end def all_missing

def any_missing(*values):
    m = False
    for v in values:
        m |= isinstance(v,Missing)
    #end for
    return m
#end def any_missing

def any_present(*values):
    return not all_missing(*values)
#end def any_present

def all_present(*values):
    return not any_missing(*values)
#end def all_present

def all_none(*values):
    n = True
    for v in values:
        n &= v is None
    #end for
    return n
#end def all_none

def any_none(*values):
    n = False
    for v in values:
        n |= v is None
    #end for
    return n
#end def any_none

# to be added to numerics.py
class GridFunction(DevBase):

    valid_boundaries = ('open','periodic')

    bc_aliases = obj(
        o='open',
        p='periodic',
        )
    for b in valid_boundaries:
        bc_aliases[b] = b
    #end for

    value_quantities = ['values']


    def __init__(self,
                 points     = missing,
                 values     = missing,
                 axes       = missing,
                 grid       = missing,
                 shift      = missing,
                 boundaries = missing,
                 shape      = 'flat',
                 copy_heavy = False,
                 copy_light = True,
                 copy       = False,
                 check      = True,
                 ):

        # initialize with no data
        self.dim        = None # grid dimension
        self.points     = None # array of function points
        self.values     = None # array of function values
        self.axes       = None # (dim,dim) array of function domain
        self.grid       = None # shape of points/values, assumed as a general grid, see http://www.xcrysden.org/doc/XSF.html
        self.shift      = None # offset of grid point nearest the origin
        self.boundaries = None # boundary conditions in each dimension
        self.shape      = None # current shape
        
        # allow empty input, otherwise require some inputs
        if all_missing(points,values,axes,shift,boundaries):
            return
        elif any_missing(values,axes):
            self._error('values and axes are required inputs')
        #end if

        # override heavy/light copying with copy input
        copy_heavy |= copy
        copy_light |= copy

        # set axes and dim
        self.set_array('axes',axes,copy=copy_light)
        self.axes = atleast_2d(self.axes)
        self.dim  = len(self.axes)

        # set values and grid
        self.set_array('values',values,copy=copy_heavy)
        if missing(grid):
            self.grid = array(self.values.shape,dtype=int)
        else:
            self.set_array('grid',grid,int,copy=copy_light)
        #end if
        self.grid = self.grid.ravel()
        self.values = self.values.ravel()

        # set shift
        if missing(shift):
            self.shift = array(self.dim*[0],dtype=float)
        else:
            self.set_array('shift',shift,copy=copy_light)
        #end if
        self.shift = self.shift.ravel()

        # set points
        if missing(points):
            ranges = []
            da = self.axes.copy()
            for d in range(self.dim):
                da[d] /= self.grid[d]
                ranges.append(arange(self.grid[d],dtype=int))
            #end for
            if self.dim==1:
                points = ranges[0]*da[0]
                points.shape = len(points),1
            else:
                #ipoints = ndgrid(*ranges) # column major/fortran
                ipoints = array(meshgrid(*ranges)) # row major/c
                ipoints.shape = self.dim,ipoints.size/self.dim
                points = dot(ipoints.T,da)
            #end if
            for d in range(self.dim):
                points[:,d] += self.shift[d]
            #end for
            self.points = points
        else:
            self.set_array('points',points,copy=copy_heavy)
            if missing(shift):
                if self.dim==1:
                    ishift = self.points.argmin()
                else:
                    ishift = dot(self.points,inv(self.axes)).sum(axis=1).argmin()
                #end if
                self.shift = self.points[ishift].copy()
            #end if
        #end if

        # set boundaries
        if missing(boundaries):
            self.set_boundaries(self.dim*'p')
        else:
            self.set_boundaries(boundaries)
        #end if

        self.shape = 'flat'
        self.set_shape(shape)

        if check:
            self.check_valid()
        #end if
    #end def __init__


    def set_array(self,name,value,dtype=float,copy=True):
        if isinstance(value,(tuple,list)):
            value = array(value,dtype=dtype)
        elif isinstance(value,ndarray):
            if copy:
                value = value.copy()
            #end if
        else:
            self._error('{0} input is invalid\nexpected tuple/list/array\nreceived type: {1}'.format(name,value.__class__.__name__))
        #end if
        self[name] = value
    #end def set_array


    def set_boundaries(self,bounds):
        if isinstance(bounds,str):
            if ' ' in bounds:
                bounds = bounds.split()
            else:
                bounds = tuple(bounds)
            #end if
        #end if
        boundaries = []
        for b in bounds:
            if b not in GridFunction.bc_aliases:
                self._error('invalid boundary input\nreceived: {0}\nexpected one of the following: {1}'.format(b,sorted(GridFunction.bc_aliases.keys())))
            #end if
            b = GridFunction.bc_aliases[b]
            boundaries.append(b)
        #end for
        self.boundaries = array(boundaries,dtype=str)
    #end def set_boundaries


    def set_shape(self,shape):
        if shape=='flat':
            self.flat_shape()
        elif shape=='full':
            self.full_shape()
        else:
            self._error('cannot set shape to: "{0}"\nvalid options are: flat,full'.format(shape))
        #end if
    #end def set_shape


    def flat_shape(self):
        if self.shape!='flat':
            npoints = self.grid.prod()
            vshape = (npoints,)
            pshape = (npoints,dim)
            for name in self.value_quantities:
                self[name].shape = vshape
            #end for
            self.points.shape = pshape
            self.shape = 'flat'
        #end if
    #end def flat_shape


    def full_shape(self):
        if self.shape!='full':
            vshape = tuple(self.grid)
            pshape = tuple(list(vshape)+[self.dim])
            for name in self.value_quantities:
                self[name].shape = vshape
            #end for
            self.points.shape = pshape
            self.shape = 'full'
        #end if
    #end def flat_shape

    def check_valid(self):
        if not isinstance(self.dim,int):
            self._error('dim is invalid\ndim must be an integer\ndim value: {0}'.format(self.dim))
        #end if
        dim = self.dim
        valid_shapes = ('flat','full')
        if self.shape not in valid_shapes:
            self._error('shape is invalid\nshape must be one of the following: {0}\nshape value: {1}'.format(valid_shapes,self.shape))
        #end if
        self.check_array('grid',self.grid,(dim,),int)
        self.check_array('axes',self.axes,(dim,dim),float)
        self.check_array('shift',self.shift,(dim,),float)
        self.check_array('boundaries',self.boundaries,(dim,),str)
        for b in self.boundaries:
            if b not in GridFunction.valid_boundaries:
                self._error('boundaries is invalid\ball boundaries must be one of the following: {0}\nboundaries value: {1}'.format(GridFunction.valid_boundaries,self.boundaries))
            #end if
        #end for
        npoints = self.grid.prod()
        if self.shape=='flat':
            vshape = (npoints,)
            pshape = (npoints,dim)
        elif self.shape=='full':
            vshape = tuple(self.grid)
            pshape = tuple(list(vshape)+[self.dim])
        #end if
        for name in self.value_quantities:
            self.check_array(name,self[name],vshape,float)
        #end for
        self.check_array('points',self.points,pshape,float)
    #end def check_valid


    def check_array(self,name,value,shape,dtype):
        if not isinstance(value,ndarray):
            self._error('{0} is invalid\n{0} must be an array\n{0} type: {1}'.format(name,value.__class__.__name__))
        elif value.shape!=shape:
            self._error('{0} is invalid\n{0} must have shape {1}\n{0} shape: {2}'.format(name,shape,value.shape))
        elif value.size>0 and not isinstance(value.ravel()[0],dtype):
            self._error('{0} is invalid\n{0} entries must be of type {1}\n{0} entries type: {2}'.format(name,dtype,value.dtype.__name__))
        #end if
    #end def check_array

    # need to fill these in for operations on values
    # also generalize all the below for StatisticalGridFunction (with error bars)
    def __add__(self,o):
        if isinstance(o, GridFunction):
            if o.check_valid:
                if self.axes == o.axes and self.grid == o.grid:
                    c = self.copy()
                    c.values +=o.values
                    if hasattr(self, 'error') and hasattr(o, 'error'):
                        c.error = sqrt(c.error**2+o.error**2)
                    #end if
                    return c
                else:
                    self._error('Added GridFunction object should have the same axes and grid')
                #end if
            else:
                self._error('Added object is not a valid instance of GridFunction')
            #end if
        else:
            self._error('Added object must be an instance of GridFunction')
        #end if
    #end def __add__

    def __sub__(self,o):
        if isinstance(o, GridFunction):
            if o.check_valid:
                if isclose(self.axes,o.axes).all() and equal(self.grid,o.grid).all():
                    c = self.copy()
                    c.values -=o.values
                    if hasattr(self, 'errors') and hasattr(o, 'errors'):
                        c.error = sqrt(c.error**2+o.error**2)
                    #end if
                    return c
                else:
                    self._error('Subtracted GridFunction object should have the same axes and grid')
                #end if
            else:
                self._error('Subtracted object is not a valid instance of GridFunction')
            #end if
        else:
            self._error('Subtracted object must be an instance of GridFunction')
        #end if
    #end def __sub__

    def __mul__(self,o):
        if isinstance(o, GridFunction) or isinstance(o, StatisticalGridFunction):
            if o.check_valid:
                if self.axes == o.axes and self.grid == o.grid:
                    c = self.copy()
                    c.values +=o.values
                    if hasattr(self, 'errors') and hasattr(o, 'errors'):
                        c.error = c.values*sqrt((self.error/self.values)**2+(o.error/o.values)**2)
                    #end if
                    return c
                else:
                    self._error('Multiplied GridFunction object should have the same axes and grid')
                #end if
            else:
                self._error('Multiplied object is not a valid instance of GridFunction')
            #end if
        else:
            self._error('Multiplied object must be an instance of GridFunction')
        #end if
    #end def __mul__
    
    def get_ghost_grid(self):
        """Returns periodic grid shape
        Converts a general grid to periodic grid by adding redundant points (ghosts)
        See: http://www.xcrysden.org/doc/XSF.html
        """
        self.full_shape()
        grid_ghost = self.grid + 1                                  
        return grid_ghost
    #end def get_ghost_grid

    def get_ghost_values(self, errors=False):
        """Returns values of a periodic grid
        See: http://www.xcrysden.org/doc/XSF.html
        """
        self.full_shape()
        grid_ghost = self.grid + 1
        values_ghost = tile(self.values, (2,2,2)) 
        values_ghost = values_ghost[0:grid_ghost[0], 0:grid_ghost[1], 0:grid_ghost[2]]
        if errors:
            errors_ghost = tile(self.errors, (2,2,2))
            errors_ghost = errors_ghost[0:grid_ghost[0], 0:grid_ghost[1], 0:grid_ghost[2]]
            return values_ghost, errors_ghost
        else:
            return values_ghost
        #end if
    #end def get_ghost_values

    def get_voxels(self):
        """Returns voxel vectors
        """
        d = []
        for i in range(0, self.dim):
            d.append(self.axes[i]/self.grid[i])
        #end for
        return d
    #end def get_voxels

    def get_scaled_grid_indexes_with_ghost(self, newgrid):
        """Returns the grid points of newgrid in grid coordinates
        Args: 
          newgrid (list[int]): 
        
        Returns:
          list[float]: 

        Say you have the grid of 80x80x80. In grid coordinate systems its points would 
        correspond to 0, 1, 2, 3, ... 78, 79 in x,y,z directions. If you want to define a 
        new grid 160x160x160, then its points inside the same volume would be 
        0, 0.5, 1, 1.5, ...., 78, 78.5, 79. 
        Since we have 'ghost' points, meaning that the grid is periodic, 
        the relation above becomes 0, 1 , 2 , 3, ... 78, 79, 80 and 
        0, 0.5, 1, 1.5, ...., 78, 78.5, 79, 79.5, 80. 
        """
        grid    = self.grid
        newgridg = array(newgrid)+1
        di = []
        for i in range(0, self.dim):
            di.append(arange(newgridg[i])*(grid[i]*1.)/newgrid[i])
        #end for
        return di
    #end def get_scaled_grid_indexes_with_ghost

    def point2grid_index(self, points):
        """Converts a point from cartesian coordinates to grid coordinates  
        """
        if points is not None:
            d       = self.get_voxels()
            xyz2g   = inv(vstack(d)) #Transform Cartesian to grid frame
            grid_indexes = []
            for p in points:
                grid_indexes.append(dot(xyz2g.T, p-self.shift))
            return array(grid_indexes)
        else:
            return points
    #end def points2grid_index

    def grid_index2point(self, grid_indexes):
        """Converts a point from grid coordinates to cartesian coordinates 
        """
        if grid_indexes is not None:
            d       = self.get_voxels()
            g2xyz   = vstack(d) #Transform Grid Frame to Cartesian
            points  =[] 
            for g in grid_indexes:
                points.append(dot(g2xyz.T, g) + self.shift)
            return array(points)
        else:
            return grid_indexes
    #end def points2grid_index
               
    def interpolate(self, newgrid, shift = None, shift_unit = 'grid_index', points_unit = None, spline_order=3):
        """Interpolates from one "regular" (i.e. not periodic) grid to another grid
        or returns the values of arbitrary points over the grid with interpolation
        
        Args:
          newgrid (int/list/ndarray): Sets new grid density depending on the type.
          shift (list/ndarray): Shifts the new grid compared to the old.
          shift_unit (str): Options are 'grid_index' and 'lattice_units'
          spline_order (int]): identical to the "order" keyword in scipy.ndimage.map_coordinates, >=0 and <=5 
        
        Returns:
          GridFunction: if newgrid is int, new grid is self.grid*newgrid 
            if newgrid is list then the new grid is newgrid 
          ndarray: if newgrid is ndarray. Then interpolation is done on the newgrid points only
        """     

        if shift is None:
            shift = [0,0,0]
        #end if

        shift_units = ['grid_index', 'lattice_units']
        if shift_unit == shift_units[0]:
            shift = {'grid_index':shift,
                     'lattice_units':self.grid_index2point([shift])[0]}
        elif shift_unit == shift_units[1]:
            shift = {'grid_index':self.point2grid_index([shift])[0],
                     'lattice_units':shift}
        else:
            self.error('Invalid shift_units: {0} are accepted'.format(shift_units))
        #end if
        
        shift = shift[shift_unit]
        
        gridg  = self.get_ghost_grid()
        has_errors = hasattr(self, 'errors')

        if has_errors:
            valueg, errorg = self.get_ghost_values(errors=True)
        else:
            valueg = self.get_ghost_values()
        #end if

        on_a_grid=False
        on_points=False
        
        if isinstance(newgrid, list) and self.dim == len(newgrid):
            on_a_grid=True
        elif isinstance(newgrid, int):
            newgrid = self.grid*newgrid
            on_a_grid=True
        elif isinstance(newgrid, ndarray) and newgrid.ndim == 2: #self.dim == newgrid.ndim:
            on_points=True
        else:
            self.error('newgrid is not a list or integer!')
        #end if

        
        if on_a_grid:
            x = self.get_scaled_grid_indexes_with_ghost(newgrid)
            X = meshgrid(*x, indexing='ij')
        elif on_points:
            if points_unit == 'grid':
                gpts = newgrid
            else:
                gpts = self.point2grid_index(newgrid)
            #end if
            
            X    = [gpts[:,i] for i in range(0,gpts.shape[1])]
        #end if
        
        for snum, s in enumerate(shift):
            X[snum]+=s
        #end if
    
        new_values = map_coordinates(valueg, X, order=spline_order, mode='wrap') #Ghosts are needed for the wrap mode
        
        if on_a_grid:
            new_values = new_values[0:-1, 0:-1, 0:-1] #Remove ghosts
       
        if has_errors:
            new_errors = map_coordinates(errorg, X, order=spline_order, mode='wrap')
            if on_a_grid:
                new_errors = new_errors[0:-1, 0:-1, 0:-1]
        #end if
        
        if on_a_grid:
            newGridFunction=dict()
            newGridFunction['shift']      = shift+self.shift
            newGridFunction['shape']      = 'full'
            newGridFunction['grid']       = newgrid
            newGridFunction['values']     = new_values
    
            newGridFunction['boundaries'] = self.boundaries
            newGridFunction['axes']       = self.axes
            #'points' are handled by GridFunction constructor

            if has_errors:
                newStatisticalGridFunction=newGridFunction
                newStatisticalGridFunction['errors'] = new_errors    
                return StatisticalGridFunction(**newStatisticalGridFunction)
            else:
                return GridFunction(**newGridFunction)
            #end if
            
        elif on_points:
            if has_errors:
                return new_values, new_errors
            else:
                return new_values
        #end if
        
    #end def interpolate

    
    def integrate(self, center=[0,0,0], r=2.0, ndr=40, darc=0.05, r_0=0.0, log_grid = False):
        """ Spherical or circular integration of the values around center
        
        Args: 
          r (float): radius
          ndr (float): number of steps on r 
          darc (float): stepsize on the arc 
          r_0 (float): starting radius for integration 
          log_grid (bool): whether to use logarithmically scaled grid on r
        
        Returns:
          [ndarray, ndarray]: r vs the integrated values
        """

        def spherical_grid(ri, darc, center):
            """Grid on the surface of a sphere
            
            Args:
              ri (float): radius
              darc (float): infinitesimal arc length 
              center (list): center of the sphere
            
            Returns:
            Flat list of points on the spherical grid in cartesian coordinates

            """
            ri         = round(ri, decimals=3)
            narcg      = int(pi*ri/darc) #Number of points on the half arc, pi is np.pi
            phi        = linspace(0, pi, narcg, endpoint = False)
            theta      = linspace(0, 2*pi, 2*narcg, endpoint = False)
            Phi, Theta = meshgrid(phi, theta)
            num_s      = Phi.shape[0]*Phi.shape[1] # Number of points on the surface

            # Convert to Cartesian
            # Meshgrids are capitalized, flat arrays are lowercase
            X = ri * sin(Phi) * cos(Theta) + center[0]
            Y = ri * sin(Phi) * sin(Theta) + center[1]
            Z = ri * cos(Phi) + center[2]

            x = X.ravel()
            y = Y.ravel()
            z = Z.ravel()
            pts = vstack((x,y,z)).T
            return pts
        #end def spherical_grid

        def circular_grid(ri, darc, center):
            """Grid on a circle
            
            Args:
              ri (float): radius
              darc (float): infinitesimal arc length 
              center (list): center of the circle
            
            Returns:
            Flat list of points on the circular grid in cartesian coordinates

            """
            ri         = round(ri, decimals=3)
            narcg      = int(pi*ri/darc)
            Phi        = linspace(0, 2*pi, narcg, endpoint = False)
 
            X = ri * sin(Phi) + center[0]
            Y = ri * cos(Phi) + center[1]
            x = X.ravel()
            y = Y.ravel()
            pts = vstack((x,y)).T
            return pts
        #end def circular_grid
        
        
        gridg  = self.get_ghost_grid()
        has_errors = hasattr(self, 'errors')

        if has_errors:
            valueg, errorg = self.get_ghost_values(errors=True)
        else:
            valueg = self.get_ghost_values()
        #end if

        dens = []
        dr = (r-r_0)/ndr
        if r_0 == 0.0:
            rlist = arange(dr,r+dr,dr)
        else:
            rlist = arange(r_0,r+dr,dr)
        #end if

        if log_grid:
            # denser r grid near center
            maxr = max(rlist)
            minr = min(rlist)
            rlist = [(maxr-minr)*(exp(i-minr)-1)/(exp(maxr-minr)-1)+minr for i in rlist]
        #end if
        
        for ri in rlist:
            if self.dim == 3:
                pts   = spherical_grid(ri, dr, center)
                omega = 4.*pi*ri**2 
            elif self.dim == 2:
                pts   = circular_grid(ri, dr, center)
                omega = 2.*pi*ri
            #end if
            
            gpts        = self.point2grid_index(pts)
            X           = [gpts[:,i] for i in range(0,gpts.shape[1])]
            value_s     = map_coordinates(valueg, X, order=3, mode='wrap')
            num_s       = len(value_s)
            avg_value_s = sum(value_s)/num_s*omega
            
            if has_errors:
                error_s     = map_coordinates(errorg, X, order=3, mode='wrap')
                avg_error_s = sqrt(sum(error_s**2))/num_s*omega 
                dens.append([ri,avg_value_s, avg_error_s])
            else:
                dens.append([ri,avg_value_s])
            #end if
        #end for

        return array(dens)
    
    #end def integrate
    
#end class GridFunction

class StatisticalGridFunction(GridFunction):
    
    value_quantities = GridFunction.value_quantities + ['errors']


    def __init__(self,**kwargs):
        errors     = kwargs.pop('errors'    ,missing)
        copy_heavy = kwargs.get('copy_heavy',True   )
        copy_light = kwargs.get('copy_light',False  )
        copy       = kwargs.get('copy'      ,False  )

        # override heavy/light copying with copy input
        copy_heavy |= copy
        copy_light |= copy

        self.errors = None
        if not missing(errors):
            self.set_array('errors',errors,copy=copy_heavy)
        #end if

        GridFunction.__init__(self,**kwargs)

        any_pres = not all_none(*self.list('points','values','axes',
                                           'shift','boundaries','errors'))
        if any_pres and any_none(self.values,self.errors,self.axes):
            self._error('values, errors, and axes are required inputs')
        #end if
    #end def __init__

#end class StatisticalGridFunction

class DensityFile(DevBase):
    
    def __init__(self,
                 name,
                 structure = None):
        filename, extension = os.path.splitext(name)

        formats = ['.xsf', '.vasp']
        if not extension in formats:
            self.error('Invalid file extension: {0} are accepted'.format(formats))
        else:
            self.name      = name
            self.extension = extension
            self.errfile   = self.get_errfile_name()
            self.structure = None
            
            if extension == formats[0]:
                self.gridf = self.xsf2gridf()
            elif extension == formats[1]:
                self.gridf = self.chgcar2gridf()
            else:
                self.gridf = None
            #end if

            # How to normalize the grid?
            # A default unit must be defined for GridFunction e/A^3, e/bohr^3, 1/A^3 etc
        #end if
    #end def __init__

    def get_errfile_name(self):
        """Search for the error file with the same name in the same directory
        
        Returns:
          str: name of the error files. Return None if not found. 
        """
        
        filename, extension = os.path.splitext(self.name)
        perr_filename = filename+'+err'+extension
        nerr_filename = filename+'-err'+extension

        errfile = None
        if os.path.exists(perr_filename):
            errfile = perr_filename
        elif os.path.exists(nerr_filename):
            errfile = nerr_filename
        #end if
        return errfile
    #end def get_errfile_name
    
    def xsf2gridf(self):
        """Convert XsfFile object to GridFunction or StatisticalGridFunction object
        
        Returns:
          StatisticalGridFunction or GridFunction object 
        """
        f = XsfFile(filepath=self.name)
        d = f.get_density()
        
        values = f.remove_ghost()
        newgrid = d.grid - 1
        if self.errfile is not None:
            ferr = XsfFile(filepath=self.errfile)
            values_err = abs(ferr.remove_ghost() - values)
            
            gridf = StatisticalGridFunction(values = values, #values_err,
                                            axes   = f.primvec,
                                            grid   = newgrid,
                                            shift  = d.corner,
                                            errors = values_err, #derr.values,
                                            #boundaries = ,
                                            shape      = 'full')
                  
        else:
            gridf = GridFunction(values = values,
                                 axes   = f.primvec,
                                 grid   = newgrid,
                                 shift  = d.corner,
                                 #boundaries = ,
                                 shape      = 'full')
        #end if

        #Convert atomic numbers to element symbols
        es = []
        for e in f.elem:
            es.append(ptable.simple_elements[e].symbol)
        #end for
        
        #Update structure
        self.structure = Structure(axes = f.primvec,
                                   elem = es,
                                   pos  = f.pos,
                                   units = 'A')
        return gridf
        # Only by reading the charge density, boundaries can't be defined.
        # f.periodicity is crystal, molecule etc.
        # Similar to other properties in structure
    #end def xsf2dens
        
    def chgcar2gridf(self):
        """Convert CHGCAR object to GridFunction or StatisticalGridFunction object
        
        Returns:
          StatisticalGridFunction or GridFunction object 
        """
        f = ChgcarFile(filepath = self.name)
        grid = f.grid
        poscar = f.poscar
        # Chgcar voxels are multiplied by number of grids
        tot_chg = f.charge_density
        spin_chg = f.spin_density
        
        has_spin = True
        if spin_chg == None:
            has_spin = False
        #end if
        #In my example ChgcarFile was not able to read spin density from any CHGCAR file
        
        if self.errfile is not None:
            ferr =  ChgcarFile(filepath = self.errfile)
            tot_chg_err = abs(ferr.charge_density - tot_chg)
            spin_chg_err = abs(ferr.spin_density  - spin_chg)
            gridf = StatisticalGridFunction(values=tot_chg,
                                             axes = poscar.axes,
                                             grid = f.grid,
                                             errors = tot_chg_err,
                                             shape = 'flat')
            if has_spin:
                gridf_spin = StatisticalGridFunction(values=spin_chg,
                                                     axes = poscar.axes,
                                                     grid = f.grid,
                                                     errors = spin_chg_err,
                                                     shape = 'flat')
                gridf = [gridf, gridf_spin]
            #end if
            
        else:
            gridf = GridFunction(values=tot_chg,
                                  axes = poscar.axes,
                                  grid = f.grid,
                                  shape = 'flat')
            if has_spin:
                gridf_spin = GridFunction(values=spin_chg,
                                          axes = poscar.axes,
                                          grid = f.grid,
                                          shape = 'flat')
                gridf = [gridf, gridf_spin]
            #end if
            
        #end if
        #Update structure
        axes = poscar.axes
        elem = poscar.elem
        pos = dot(poscar.pos, axes)
        self.structure = Structure(axes = axes,
                                   elem = elem,
                                   pos  = pos,
                                   units= 'A')
        return gridf
    #end def chgcar2dens
    
    def get_grid_function(self):
        return self.gridf
    #end def get_grid_function

    def get_structure(self):
        return self.structure
    #end def get_structure

    def has_multiple_gridfunctions(self):
        if isinstance(self.gridf, list) and len(self.gridf) > 1:
            return True
        else:
            return False
    #end def has_multiple_gridfunctions

    def has_error_gridfunction(self):
        gf = self.get_grid_function()
        has_error_gridf = []
        if self.has_multiple_gridfunction():
            for g in gf:
                if isinstance(g, StatisticalGridFunction):
                    has_error_gridf.append(True)
                else:
                    has_error_gridf.append(False)
                #end if
            #end force
        else:
            if isinstance(gf, StatisticalGridFunction):
                has_error_gridf.append(True)
            else:
                has_error_gridf.append(False)
            #end if
        #end if

        if has_error_gridf.all():
            return True
        elif not has_error_gridf.any():
            return False
        else:
            self.error('Multiple Grid Functions which are not all StatisticalGridFunction or all GridFunction')
        #end if
    #end def has_error_gridfunction
    
#end def class DensityFile

# to be added to new quantity_analyzers.py
class DensityAnalyzer(DevBase):

    def __init__(self,
                 name,
                 density   = None, # file name, fileio class, or GridFunction object
                 structure = None, # Nexus structure class, optional
                 ):
        self.name = name
        if density is None:
            density = name
        #end if
        self.add_density(density,structure)
    #end def __init__

    def add_density(self,dens, structure = None, check_error_file = False):
        """Add new density to the DensityAnalyzer object
        
        Args:
          dens (str/DensityFile/GridFunction): Density to be added. 
          structure (Structure): Structure object to be added. Ignored if dens is type str or DensityFile.
          check_error_file (bool): Whether to look for an error file
        """
        if isinstance(dens,str): 
            d              = DensityFile(dens)
            self.density   = d.get_grid_function()
            self.structure = d.get_structure()
        elif isinstance(dens, DensityFile):
            d              = dens
            self.density   = d.get_grid_function()
            self.structure = d.get_structure()
        elif isinstance(dens, GridFunction):
            self.density   = dens
            self.structure = structure
            if self.structure == None:
                self.error('No structure is given in GridFunction')
            #end if
        else:
            self.not_implemented()
        #end if
         
    #end def add_density
    
    def write_density(self, filename='density.xsf'):
        """Write density to a file. Write format is guessed from the filename. 
        Args:
          filename (str): Output density filename. 
        
        Returns:
           File with the filename in the current directory. 
        """
        density = self.density
        if isinstance(density, StatisticalGridFunction):
            self.warn('Error of the density is not printed')
        #end if
        
        if filename is not None:
            name, extension = os.path.splitext(filename)
        #end if

        formats = ['.xsf', '.vasp', '.chgcar']
        if not extension in formats:
            self.error('Invalid file extension: Only {0} are accepted'.format(formats))
        #end if
                
        density   = self.density
        structure = self.structure
        xsf = XsfFile()
        xsf.incorporate_structure(structure)
        xsf.add_density(density.axes, density.values, name='density', centered=False, add_ghost=True)
        
        if extension == formats[0]:
            xsf.write(filename)
        elif extension == formats[1] or extension == formats[2]:
            chgcar = ChgcarFile()
            xsf.add_density(density.axes, density.values, name='density', centered=False, add_ghost=False)
            chgcar.incorporate_xsf(xsf)
            chgcar.write(filename)
        else:
            self.error()
            self.not_implemented()
        #end if
    #end def write_density

    def planar_density(self, plane=[1,0,0], plot=True, return_plot=False):
        """Plots the density on a plane. Currently only works with simple [100] family planes
        
        Args:
          plane (list): Miller indices of the plane (must be positive).
          plot (bool): Whether to plot the density. 
          return_plot (bool): Whether to return the matplotlib object for further adjustments elsewhere
        
        Returns:
          None: When plot is True but return_plot is False. Density is plotted on x-windows.
          Matplotlib.pylot: When plot and return_plot are both True.
          [numpy.meshgrid, ndarray]: When plot is False. Return raw grid and values. 
        
        """
        dens = self.density
        plane = array(plane)
        grid = dens.grid
        gridg = dens.get_ghost_grid()
        plane_grid_ghost = gridg-dens.grid*plane
        
        #plane_grid_ghost = plane_grid
        x = []
        for gm in plane_grid_ghost:
            x.append(arange(gm))
        #end for

        X = meshgrid(*x, indexing='ij')
        x = X[0].ravel()
        y = X[1].ravel()
        z = X[2].ravel()
        pts = vstack((x,y,z)).T
        planar_dens = dens.interpolate(pts,points_unit='grid',spline_order=0) #[0]
	if len(planar_dens) == 2:
	    planar_dens = planar_dens[0]
	#end if
        plane_grid_ghost = filter(lambda a: a != 1, plane_grid_ghost)
        planar_dens = squeeze(planar_dens.reshape(plane_grid_ghost))
        planar_shape = planar_dens.shape
        
        abc = norm(self.structure.axes, axis=1)
        planar_ab = abc-abc*plane
        planar_ab = planar_ab[planar_ab!=0.0]

        planar_x = linspace(0,planar_ab[0],num=planar_shape[0])
        planar_y = linspace(0,planar_ab[1],num=planar_shape[1])
        planar_grid = meshgrid(planar_x,planar_y)
        
        if plot:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8,6))
	    plot_obj = plt.contourf(planar_x,planar_y,planar_dens, cmap=plt.cm.coolwarm)
            cbar     = plt.colorbar(plot_obj)
            plt.xlabel(r'$L_x (\AA)$')
            plt.ylabel(r'$L_y (\AA)$')
            if return_plot:
                return plot_obj
            else:
                plt.show()
            #end if
        else:
            return [planar_grid, planar_dens]
        #end if
        
    #end def planar_density
    
        
    def interpolate(self, newgrid, shift= None, shift_unit='grid_index', spline_order = 3, qmcpack_to_qe = False):
        """Interpolates from one "regular" (i.e. not periodic) grid to another grid
        or returns the values of arbitrary points over the grid with interpolation
        
        Args:
          newgrid (int/list/ndarray): Sets new grid density depending on the type.
          shift (list/ndarray): Shifts the new grid compared to the old.
          shift_unit (str): Options are 'grid_index' and 'lattice_units'
          spline_order (int]): identical to the "order" keyword in scipy.ndimage.map_coordinates, >=0 and <=5 
          qmcpack_to_qe (bool): Default accumulated qmcpack density is shifted. Apply a shift and rescale the density (not complete). 

        Returns:
          DensityAnalyzer: if newgrid is int, new grid is self.grid*newgrid 
            if newgrid is list then the new grid is newgrid 
        """
        
	if qmcpack_to_qe:
            shift = [-0.5, -0.5, -0.5]
            # rescaling factor should be added here -- Kayahan
        elif shift is None:
            shift = [0,0,0]
        #end if

        dens            = self.density
        name, extension = os.path.splitext(self.name)

        if isinstance(newgrid, list):
            newname = name+'_'+'_'.join(str(x) for x in newgrid)+extension
        elif isinstance(newgrid, int):
            newname = name+'_'+str(newgrid)+extension
            newgrid = self.grid*newgrid
        else:
            self.error('newgrid must be a list or int')
        #end if
    
        newdens         = dens.interpolate(newgrid, shift=shift, shift_unit=shift_unit, spline_order=spline_order)
        newdaobj        = DensityAnalyzer(newname, newdens, self.structure)
        
        return newdaobj
        #end if
    #end def interpolate

    def equal_grid(self, da_obj): 
        """Returns true if both DensityAnalyzer objects have identical grids
        """
        grid_a = self.density.grid
        grid_b = da_obj.density.grid
        if all(grid_a == grid_b):
            return True
        else:
            return False
        #end if
    #end def equal_grid
    
    def extrapolate(self, da_obj, self_density_source='dmc', method='linear'): 
        """Mixed density estimator from VMC and DMC densities. See Eqn. 3.54 and 3.55 of Rev. Mod. Phys. 2001, 73(1) 33-83 for methods. 
        
        Args:
          da_obj: The other DensityAnalyzer object used in the extrapolation
          self_density_source: Self DensityAnalyzer object is from dmc or vmc?
          method: Linear or quadratic. Should usually result identical.
        
        Returns:
          DensityAnalyzer: Extrapolated DensityAnalyzer object
        """
        if not isinstance(da_obj, DensityAnalyzer):
            self.error('{0} is not an instance of DensityAnalyzer'.format(density))
        #end if
        
        if not self.equal_grid(da_obj):
            self.error('{0} and {1} do not share the same grid'.format(self, density))
        else:
            da_ext = self.copy() # assume identical structures
            da_ext.name = da_ext.name+'_ext'
        #end if
        
        if self_density_source == 'vmc':
            d_vmc = self.density.values
            d_dmc = da_obj.density.values
        elif self_density_source == 'dmc':
            d_vmc = da_obj.density.values
            d_dmc = self.density.values
        else:
            self.error('density_source "{0}" should be vmc or dmc'.format(self_density_source))
        #end if

        if method == 'linear':
            da_ext.density.values = 2*d_dmc - d_vmc
        elif method == 'quad':
            da_ext.density.values = d_dmc**2/d_vmc
        else:
            self.error('{0} method is not defined'.format(method))
        #end if

        return da_ext
    #end def extrapolate
        
    def integrate(self, center, r=2.0, ndr=40, darc=0.05, r_0 = 0,
                  log_grid = False, plot = True, return_plot=False):
        """ Spherical or circular integration of the values around center
        
        Args: 
          center (int, list[float], str): If int then integration is made around self.structure.pos[center].
            If list[float], then self.dim must be equal to len(list). Integration is made around center. 
            If str, then integration is made around self.structure.elem == center and averaged. 
          r (float): Radius
          ndr (float): Number of steps on r
          darc (float): Stepsize on the arc.
          r_0 (float): Starting radius for integration 
          log_grid (bool): Whether to use logarithmically scaled grid on r
          plot (bool): Whether to plot the density
          return_plot (bool): Whether to return the matplotlib object for further adjustments elsewhere
        
        Returns:
          None: When plot is True but return_plot is False. Density is plotted on x-windows.
          Matplotlib.pylot: When plot and return_plot are both True.
          [ndarray, ndarray]: r vs the integrated values
        """
        
        dens = self.density
        single_center = True
        center_name = None
        if isinstance(center, str):
            single_center = False
            center_name = center
            # Try species defined with atom names
            centers = where(self.structure.elem == center)[0]
            
            if len(centers) == 0:
                # Then try species defined with atomic numbers
                element = ptable[center]
                center = element.atomic_number
                centers = where(self.structure.elem == center)[0]

                if len(centers) == 0:
                    self.error('{0} is not in structure'.format(atom))
                #end if
            #end if
            center = self.structure.pos[centers]
            atom = center
        elif isinstance(center, int):
            center_name = center
            center = self.structure.pos[center]
        elif isinstance(center, list):
            center_name = list
            center = [self.structure.pos[x] for x in center]
            single_center = False
        else:
            self.error('center format is incorrect!')
        #end if

        if single_center:
            results = [dens.integrate(center, r, ndr, darc, r_0,
                                      log_grid=log_grid)]
        else:
            results = []
            for c in center:
                results.append(dens.integrate(c, r, ndr, darc, r_0,
                                              log_grid=log_grid))
            #end for
        #end if

        has_errors = (results[0].shape[1] == 3)

        avg_results = zeros(results[0].shape)
        avg_results[:,0]= results[0][:,0] #Copy rlist to average results
        nresults = len(results)
        
        for rs in results:
            avg_results[:,1]+=rs[:,1]/nresults #mean 
            if has_errors:
                avg_results[:,2]+=rs[:,1]**2/nresults #uncertainty
            #end if
        #end if
        

        if plot:
            import matplotlib.pyplot as plt
            x = avg_results[:,0]
            y = avg_results[:,1]
            plt.xlabel('Radius')
            plt.ylabel('Charge density')
            plt.title(self.name)
            plot_obj = []
            if has_errors:
                plot_obj = plt.errorbar(x, y, yerr = avg_results[:,2], label=center_name, fmt='-s')
            else:
                plot_obj = plt.plot(x,y, '-s', label=center_name)
            #end if
            if return_plot:
                return plot_obj
            else:
                plt.show()
            #end if
        else:
            return avg_results
        #end if
        
    #end def integrate
        
    # Jaron's notes
    # determine file type and read w/ fileio classes
    # probably good to define DensityFile interface in fileio
        
        #if isinstance(dens,DensityFile):
        #    dens = dens.get_grid_function()
        #    struct = dens.get_structure()
        #elif not isinstance(dens,GridFunction):
        #    self.error('cannot add density\ndensity must be represented by a GridFunction object')
        #end if
        #self.density = dens
        #self.structure = struct
    #end def add_density
#end class DensityAnalyzer


if __name__=='__main__':

    print
    print 'empty grid function'
    gf = GridFunction()
    print gf

    print 
    print 'simple 1D grid function'
    gf = GridFunction(values=[3,6,2,4,9],axes=[5])
    print gf

    print 
    print 'simple 2D grid function (flat)'
    gf = GridFunction(values=[[0,1,2],[3,4,5],[6,7,8]],axes=[[3,0],[0,3]])
    print gf
    print 'simple 2D grid function (full)'
    gf.full_shape()
    print gf

    print 
    print 'simple 3D grid function (flat)'
    gf = GridFunction(values=27*[3.14],grid=(3,3,3),axes=[[3,0,0],[0,3,0],[0,0,3]])
    print gf
    print 'simple 3D grid function (full)'
    gf.full_shape()
    print gf

    print 
    print 'simple 3D stat grid function (flat)'
    gf = StatisticalGridFunction(values=27*[3.14],errors=27*[0.1],
                                 grid=(3,3,3),axes=[[3,0,0],[0,3,0],[0,0,3]])
    print gf

    print 'interpolation of simple 3D stat grid function (flat)'
    gf2 = StatisticalGridFunction(values=6000*[1.0],errors=6000*[0.1],
                                  grid=(30,20,10),axes=[[3,0,0],[0,4,0],[0,0,5]])
    gf_int = gf2.interpolate(2, shift=[0.5, 0.5, 0.5])
    print gf_int
    
    print 'spherical integration of simple 3D stat grid function'
    result = gf2.integrate(r=1.0)
    print result
    
    print 'spherical integration of around all oxygen atoms averaged in an .xsf file'
    kk = DensityAnalyzer('./chg.dens.xsf')
    print 'using log_grid'
    kk.integrate('O', log_grid = True)
    print 'without using log_grid'
    kk.integrate('O', log_grid = False)

    print 'spherical integration of around all cobalt atoms averaged in a .vasp file'
    kk = DensityAnalyzer('./CHGCAR.vasp')
    print 'using log_grid'
    kk.integrate('Co', log_grid = True)
    print 'without using log_grid'
    kk.integrate('Co', log_grid = False)
    

    print 'spherical integration of around the first atom in a .vasp file'
    kk = DensityAnalyzer('./CHGCAR.vasp')
    print 'using log_grid'
    kk.integrate(0, log_grid = True)
    print 'without using log_grid'
    kk.integrate(0, log_grid = False)

    print 'finished'
    
#end if

