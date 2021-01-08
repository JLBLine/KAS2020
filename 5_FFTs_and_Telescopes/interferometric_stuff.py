# import matplotlib
# matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

def add_colourbar(fig=None,ax=None,im=None,label=False,top=False):
    """Adds a colorbar to a specific axis (ax) on a given figure (fig),
    using an image (im) output of imshow. Optionally add a label (label) to the
    colorbar. Defaults to adding the colorbar on the right on the axes, change
    top=True to plot the colorbar on top of the axes instead"""
    divider = make_axes_locatable(ax)
    if top == True:
        cax = divider.append_axes("top", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax = cax,orientation='horizontal')
        cax.xaxis.set_ticks_position('top')
        cax.xaxis.set_label_position('top')
    else:
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax = cax)
    if label:
        cbar.set_label(label)


class UVData_FromImage():

    def __init__(self,image):
        """A class to take a 2D image, and fourier transform to make simulated
        u,v data. Contains a number of attributes to apply various dish or
        interferometric observational effects, to demonstrate the effects
        of the instrument on image reconstruction"""

        ##TODO check data is 2D array, and has an odd NSIDE
        self.nside = image.shape[0]
        self.image = image
        ##setup default values that are useful for later attributes
        self.masked_uvdata = False
        self.uvdata = False
        self.mwa_lat_rad = -26.7033194444*(np.pi/180.0)
        ##Speed of light m/s
        self.VELC = 299792458.0
        self.default_img_size = 10*(np.pi/180.0)

    def create_uvdata(self):
        """Perform a 2D FFT on the input image, to create uv data"""

        uvdata = np.fft.fft2(self.image)
        uvdata = np.fft.fftshift(uvdata)
        self.uvdata = uvdata

        return self.uvdata
#
    def _make_circle(self, radius, inverse=False):
        """Makes a 2D array of the same dimensions of the input image, with
        a circle of a given radius. Defaults to setting array to 1.0 inside
        the radius and 0.0 outside, optionally set inverse=True to set all to
        0.0 inside radius and 1.0 outside"""
        if radius > self.nside / 2:
            sys.exit('Cannot apply a circular mask with radius greater \
                      than half the span of the image - try with a smaller \
                      radius. Exiting now to save future pain')
        else:
            cent_pix = int(np.floor(self.nside/2))
            coord_range = np.arange(self.nside) - cent_pix
            x_mesh, y_mesh = np.meshgrid(coord_range, coord_range)

            circ_mask = np.zeros(x_mesh.shape)

            if inverse:
                circ_mask[np.sqrt(x_mesh**2 + y_mesh**2) > radius] = 1.0

            else:
                circ_mask[np.sqrt(x_mesh**2 + y_mesh**2) <= radius] = 1.0

        return circ_mask

    def apply_circ_mask(self, radius):
        """Apply a cirular mask of radius in pixel units (everything
        outside the circle is set to zero)"""
        circ_mask = self._make_circle(radius)
        self.masked_uvdata = self.uvdata*circ_mask

        return self.masked_uvdata
#
    def apply_inverse_circ_mask(self, radius):
        """Apply an inverse cirular mask of radius in pixel units (everything
        inside the circle is set to zero)"""
        circ_mask = self._make_circle(radius, inverse=True)
        self.masked_uvdata = self.uvdata*circ_mask

        return self.masked_uvdata

    def _image_uvdata(self, uvdata):
        """Performs an inverse FFT on the 2D array (uvdata), and returns
        the real component of the resultant image."""
        uvdata = np.fft.ifftshift(uvdata)
        image_data = np.fft.ifft2(uvdata)
        return np.real(image_data)
        # return np.fft.ifftshift(image_data)

    def _image_unmasked_uvdata(self):
        """Little hidden attribute to check that the imaging method
        returns the original image, meaning the whole thing is consistent"""
        if type(self.uvdata) == bool:
            sys.exit('Must run .create_uvdata to be able to run ._image_unmasked_uvdata')

        else:
            return self._image_uvdata(self.uvdata)

    def _enh2xyz(self, east, north, height,latitiude=False):
        """Takes east, north, height and converts to local X,Y,Z. Latitiude
        defaults to MWA"""

        if latitiude:
            pass
        else:
            latitiude = self.mwa_lat_rad

        sl = np.sin(latitiude)
        cl = np.cos(latitiude)
        X = -north*sl + height*cl
        Y = east
        Z = north*cl + height*sl
        return X,Y,Z

    def _get_uvw_freq(self, x_length=None,y_length=None,z_length=None,
                      frequency=200e+6, dec=False, ha=0.0):
        """Takes the baseline length in meters and uses the frequency to
        calculate u,v,w. All angles should be in radians"""

        if dec:
            pass
        else:
            dec = self.mwa_lat_rad

        scale = frequency / self.VELC

        x = x_length * scale
        y = y_length * scale
        z = z_length * scale

        u = np.sin(ha)*x + np.cos(ha)*y
        v = -np.sin(dec)*np.cos(ha)*x + np.sin(dec)*np.sin(ha)*y + np.cos(dec)*z
        w = np.cos(dec)*np.cos(ha)*x - np.cos(dec)*np.sin(ha)*y + np.sin(dec)*z

        return u,v,w

    def _make_uvw_mask(self):
        """Given a list of calculated u,v data points in self.us, self.vs,
        use self.u_range and self.nside to create a uvw mask. Grids the desired
        u,v points to the closest grid point to create a mask"""

        mask = np.zeros((self.nside,self.nside))

        for u,v in zip(self.us, self.vs):

            if abs(u) > max(self.u_range) or abs(v) > max(self.u_range):
                pass
            else:

                u_offs = np.abs(self.u_range - u)
                ##Find out where in the gridded u coords the current u lives;
                ##This is a boolean array of length len(u_offs)
                ##Find the index so we can access the correct entry in the container
                u_ind = np.where(u_offs < self.u_reso/2.0)[0]

                ##Use the numpy abs because it's faster (np_abs)
                v_offs = np.abs(self.u_range - v)
                v_ind = np.where(v_offs < self.u_reso/2.0)[0]

                mask[v_ind,u_ind] = 1.0

        return mask + mask[::-1,::-1]


    def apply_enh_mask(self, east, north, height, frequency=200e+6,
                       img_size=False, dec=False, ha=0.0):
        """Given a list of east, north, height coords (m), calculate the u,v,w
        coverage, create a visibility mask, and apply to the u,v data. As
        u,v coords are frequency dependent, defaults to 200MHz, but can be
        changed. Also have to know the size of the original image on the sky,
        which defaults to 10 deg, but can be changed with img_size. Defaults
        to setting phase centre to dec=MWA latitude, hour angle of 0.0. Can
        be controlled using dec, ha (rad)"""

        if dec:
            pass
        else:
            dec = self.mwa_lat_rad

        ##to apply a u,v,w mask, we need to know how big on the sky our
        ##image actually is
        if img_size:
            pass
        else:
            img_size = self.default_img_size



        num_ants = len(east)

        ##Convert e,n,h to X,Y,Z
        X, Y, Z = self._enh2xyz(east, north, height)

        x_lengths = []
        y_lengths = []
        z_lengths = []

        ##Go through all antenna combos to make baselines
        for ant1 in np.arange(num_ants-1):
            for ant2 in np.arange(ant1+1,num_ants):

                x_lengths.append(X[ant2] - X[ant1])
                y_lengths.append(Y[ant2] - Y[ant1])
                z_lengths.append(Z[ant2] - Z[ant1])

        ##use x,y,z differences to create u,v,w
        us, vs, ws = self._get_uvw_freq(x_length=np.array(x_lengths),
                                   y_length=np.array(y_lengths),
                                   z_length=np.array(z_lengths),
                                   frequency=frequency, dec=dec, ha=ha)

        self.us = us
        self.vs = vs
        self.ws = ws

        ##Set angular extent of image to img_size
        ##fourier dual of u,v,w is l,m,n, which are cosine angles
        ##so extent is sin(angle)
        l_extent = np.sin(img_size)
        l_res = l_extent / self.nside

        ##figure out u coords
        self.u_range = np.fft.fftshift(np.fft.fftfreq(self.nside,l_res))
        self.u_reso = self.u_range[1] - self.u_range[0]

        uvw_mask = self._make_uvw_mask()

        self.masked_uvdata = uvw_mask*self.uvdata
        return uvw_mask, self.masked_uvdata

    def image_masked_uvdata(self):
        """Return an image of the current masked data set"""
        if type(self.masked_uvdata) == bool:
            sys.exit('No mask has been applied to uvdata, so cannot \
                     run .image_masked_data. Try running either:\n \
                     \t .apply_circ_mask\n \
                     \t .apply_inverse_circ_mask\n \
                     \t .apply_enh_mask\n \
                     Exiting now')

        else:
            return self._image_uvdata(self.masked_uvdata)


def plot_uvdata_left_image_right(uvdata, image, title=False, img_cmap='Blues'):
    """Plot complex uvdata on the left, real image data on the right.
    If title is set, save the image to that title. Use img_cmap to change
    the default colour map for the image (default is Blues)"""
    fig,axs = plt.subplots(1,2,figsize=(8,4))

    im0 = axs[0].imshow(np.abs(uvdata),cmap='Reds')
    im1 = axs[1].imshow(image,cmap=img_cmap)

    ims = [im0,im1]
    titles = ['Amplitude $u,v$ data','Sky image']

    for ind,ax in enumerate(axs):
        add_colourbar(fig=fig,ax=ax,im=ims[ind])
        ax.set_title(titles[ind])
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    if title:
        fig.savefig(title,bbox_inches='tight')
        plt.close()

def plot_image_left_uvdata_right(image, uvdata, title=False, img_cmap='Blues'):
    """Plot real image data on the left, complex uvdata on the right.
    If title is set, save the image to that title. Use img_cmap to change
    the default colour map for the image (default is Blues)"""
    fig,axs = plt.subplots(1,2,figsize=(8,4))

    im0 = axs[0].imshow(image,cmap=img_cmap)
    im1 = axs[1].imshow(np.abs(uvdata),cmap='Reds')

    ims = [im0,im1]
    titles = ['Sky image', 'Amplitude $u,v$ data']

    for ind,ax in enumerate(axs):
        add_colourbar(fig=fig,ax=ax,im=ims[ind])
        ax.set_title(titles[ind])
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    if title:
        fig.savefig(title,bbox_inches='tight')
        plt.close()
