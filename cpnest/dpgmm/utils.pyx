from __future__ import division
import numpy as np
cimport numpy as np
from libc.math cimport sqrt,atan2,acos,sin,cos

#---------
# utilities
#---------

cpdef  tuple eq2ang(double ra, double dec):
    """
    convert equatorial ra,dec in radians to angular theta, phi in radians
    parameters
    ----------
    ra: scalar or array
        Right ascension in radians
    dec: scalar or array
        Declination in radians
    returns
    -------
    theta,phi: tuple
        theta = pi/2-dec*D2R # in [0,pi]
        phi   = ra*D2R       # in [0,2*pi]
    """
    cdef double phi = ra
    cdef double theta = np.pi/2. - dec
    return theta, phi

cpdef  tuple ang2eq(double theta, double phi):
    """
    convert angular theta, phi in radians to equatorial ra,dec in radians
    ra = phi*R2D            # [0,360]
    dec = (pi/2-theta)*R2D  # [-90,90]
    parameters
    ----------
    theta: scalar or array
        angular theta in radians
    phi: scalar or array
        angular phi in radians
    returns
    -------
    ra,dec: tuple
        ra  = phi*R2D          # in [0,360]
        dec = (pi/2-theta)*R2D # in [-90,90]
    """
    
    cdef double ra = phi
    cdef double dec = np.pi/2. - theta
    return ra, dec

cpdef  tuple cartesian_to_spherical(np.ndarray[np.float64_t, ndim=1] vector):
    """Convert the Cartesian vector [x, y, z] to spherical coordinates [r, theta, phi].

    The parameter r is the radial distance, theta is the polar angle, and phi is the azimuth.


    @param vector:  The Cartesian vector [x, y, z].
    @type vector:   numpy rank-1, 3D array
    @return:        The spherical coordinate vector [r, theta, phi].
    @rtype:         numpy rank-1, 3D array
    """

    # The radial distance.
    cdef unsigned int i
    cdef double r 
    for i in range(3):
        r += vector[i]*vector[i]
    r = sqrt(r)
    # Unit vector.
    cdef np.ndarray[np.float64_t, ndim=1] unit = vector / r

    # The polar angle.
    cdef double theta = acos(unit[2])

    # The azimuth.
    cdef double phi = atan2(unit[1], unit[0])

    # Return the spherical coordinate vector.
    return r, theta, phi


cpdef  np.ndarray[np.float64_t, ndim=1] spherical_to_cartesian(np.ndarray[np.float64_t, ndim=1] spherical_vect):
    """Convert the spherical coordinate vector [r, theta, phi] to the Cartesian vector [x, y, z].

    The parameter r is the radial distance, theta is the polar angle, and phi is the azimuth.


    @param spherical_vect:  The spherical coordinate vector [r, theta, phi].
    @type spherical_vect:   3D array or list
    @param cart_vect:       The Cartesian vector [x, y, z].
    @type cart_vect:        3D array or list
    """
    cdef np.ndarray[np.float64_t, ndim=1] cart_vect = np.zeros(3)
    # Trig alias.
    cdef double sin_theta = sin(spherical_vect[1])

    # The vector.
    cart_vect[0] = spherical_vect[0] * cos(spherical_vect[2]) * sin_theta
    cart_vect[1] = spherical_vect[0] * sin(spherical_vect[2]) * sin_theta
    cart_vect[2] = spherical_vect[0] * cos(spherical_vect[1])
    return cart_vect

cpdef  np.ndarray[np.float64_t, ndim=1] celestial_to_cartesian(np.ndarray[np.float64_t, ndim=1] celestial_vect):
    """Convert the spherical coordinate vector [r, dec, ra] to the Cartesian vector [x, y, z]."""
    celestial_vect[1]=np.pi/2. - celestial_vect[1]
    return spherical_to_cartesian(celestial_vect)

cpdef  np.ndarray[np.float64_t, ndim=1] cartesian_to_celestial(np.ndarray[np.float64_t, ndim=1] cartesian_vect):
    """Convert the Cartesian vector [x, y, z] to the celestial coordinate vector [r, dec, ra]."""
    spherical_vect = cartesian_to_spherical(cartesian_vect)
    spherical_vect[1]=np.pi/2. - spherical_vect[1]
    return spherical_vect

cpdef  double Jacobian(np.ndarray[np.float64_t, ndim=1] cartesian_vect):
    d = sqrt(cartesian_vect.dot(cartesian_vect))
    d_sin_theta = sqrt(cartesian_vect[:-1].dot(cartesian_vect[:-1]))
    return d*d_sin_theta
