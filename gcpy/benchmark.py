""" Specific utilities re-factored from the benchmarking utilities. """

import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.backends.backend_pdf import PdfPages
import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes  # for assertion
from .plot import WhGrYlRd, add_latlon_ticks
from .util import safe_div

# change default fontsize (globally)
# http://matplotlib.org/users/customizing.html
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.titlesize'] = 14

# Color map for absolute magnitude
cmap_abs = WhGrYlRd  # for plotting absolute magnitude

# Color map for difference
cmap_diff = 'RdBu_r'


def plot_layer(dr, ax, title='', unit='', diff=False, ratio=False,
               vmin=None, vmax=None, shrink=0.8):
    '''Plot 2D DataArray as a lat-lon layer

    Parameters
    ----------
    dr : xarray.DataArray
        Dimension should be [lat, lon]

    ax : Cartopy GeoAxes object with PlateCarree projection
        Axis on which to plot this figure

    title : string, optional
        Title of the figure

    unit : string, optional
        Unit shown near the colorbar

    diff : Boolean, optional
        Switch to the color scale and range for difference plots.

    ratio : Boolean, optional
        Switch to the color scale and range for ratio plots.


    shrink : float, optional
        Horizontal extent of the colorbar (1.0 = width of plot)
    '''

    assert isinstance(ax, GeoAxes), (
           "Input axis must be cartopy GeoAxes! "
           "Can be created by: \n"
           "plt.axes(projection=ccrs.PlateCarree()) \n or \n"
           "plt.subplots(n, m, subplot_kw={'projection': ccrs.PlateCarree()})"
           )
    assert ax.projection == ccrs.PlateCarree(), (
           'must use PlateCarree projection'
           )

    # Get parent figure from the axis object
    fig = ax.figure 

    # Default min and max values
    if vmax == None and vmin == None:
        if diff:
            vmax = np.max(np.abs(dr.values))
            vmin = -vmax
        elif ratio:
            vmax = 2.0
            vmin = 0.0
        else:
            vmax = np.max(dr.values)
            vmin = 0.0

    # Pick the colortable
    if diff or ratio:
        cmap = cmap_diff
    else:
        cmap = cmap_abs

    # imshow() is 6x faster than pcolormesh(), but with some limitations:
    # - only works with PlateCarree projection
    # - the left map boundary can't be smaller than -180,
    #   so the leftmost box (-182.5 for 4x5 grid) is slightly out of the map
    im = dr.plot.imshow(ax=ax, vmin=vmin, vmax=vmax, cmap=cmap,
                        transform=ccrs.PlateCarree(),
                        add_colorbar=False)

    # Add the colorbar for difference plots.  
    # Set the range symmetrically between vmin and vmax
    if diff:
        cb = fig.colorbar(im, ax=ax, shrink=shrink,
                           orientation='horizontal', pad=0.10,
                           ticks=[vmin, vmin/2.0, 0, vmax/2.0, vmax ],
                           format='%.2e' )


    # Add the colorbar for ratio plots.
    # Set the range to 0.0 and 2.0, with 1.0 in the center
    elif ratio:
        cb = fig.colorbar(im, ax=ax, shrink=shrink,
                          orientation='horizontal', pad=0.10,
                          ticks=[0.0, 0.5, 0.9, 1.1, 1.5, 2.0 ],
                          format='%.1f' )

    # Add the colorbar for absolute magnitude plots
    # Use the tick locator to plot relevant tick intervals
    else:
        cb = fig.colorbar(im, ax=ax, shrink=shrink,
                          orientation='horizontal', pad=0.1 )
        cb.locator = ticker.MaxNLocator(nbins=5)
        cb.update_ticks()
    
    # Add the unit string
    cb.set_label(unit)

    # Set the tick length and label size
    cb.ax.tick_params(length=3, labelsize=10)

    # array automatically sets a title which might contain dimension info.
    # surpress it or use user-provided title
    ax.set_title(title)

    # Add coastlines
    ax.coastlines()

    # Add ticks and gridlines
    add_latlon_ticks(ax) 


def plot_zonal(dr, ax, title='', unit='', diff=False, ratio=False,
               vmin=None, vmax=None, shrink=0.8):
    '''Plot 2D DataArray as a zonal profile

    Parameters
    ----------
    dr : xarray.DataArray
        dimension should be [lev, lat]

    ax : matplotlib axes object
        Axis on which to plot this figure

    title : string, optional
        Title of the figure

    unit : string, optional
        Unit shown near the colorbar

    diff : Boolean, optional
        Switch to the color scale for difference plot
     
    ratio : Boolean, optional
        Switch to the color scale and range for ratio plots.

    shrink : float, optional
        Horizontal extent of the colorbar (1.0 = width of plot)
    '''

    # Assume global field from 90S to 90N
    xtick_positions = np.array([-90, -60, -30, 0, 30, 60, 90])
    xticklabels = ['90$\degree$S',
                   '60$\degree$S',
                   '30$\degree$S',
                   '0$\degree$',
                   '30$\degree$N',
                   '60$\degree$N',
                   '90$\degree$N'
                   ]

    # Tet the parent figure
    fig = ax.figure 

    # Pick the color map 
    if diff or ratio:
        cmap = cmap_diff
    else:
        cmap = cmap_abs

    # Define default min and max values 
    if vmin == None and vmax == None:
        if diff:
            vmax = np.max(np.abs(dr.values))
            vmin = -vmax
        elif ratio:
            vmax = 2.0
            vmin = 0.0
        else:
            vmax = np.max(dr.values)
            vmin = np.min(dr.values)

    # imshow() is 6x faster than pcolormesh(), but with some limitations:
    # - only works with PlateCarree projection
    # - the left map boundary can't be smaller than -180,
    #   so the leftmost box (-182.5 for 4x5 grid) is slightly out of the map
    im = dr.plot.imshow(ax=ax, vmin=vmin, vmax=vmax, 
                        cmap=cmap, add_colorbar=False)

    # the ratio of x-unit/y-unit in screen-space
    # 'auto' fills current figure with data without changing the figure size
    ax.set_aspect('auto')

    # Top-of-plot label
    ax.set_title(title)

    # X-axis ticks
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(xticklabels)

    # Y-axis ticks and label
    ax.set_xlabel('')
    ax.set_ylabel('Level')

    # NOTE: The surface has a hybrid eta coordinate of 1.0 and the
    # atmosphere top has a hybrid eta coordinate of 0.0.  If we don't
    # flip the Y-axis, then the surface will be plotted at the top
    # of the plot. (bmy, 3/7/18)
    ax.invert_yaxis()
    
    # Add the colorbar for difference plots.  
    # Set the range symmetrically between vmin and vmax
    if diff:
        cb = fig.colorbar(im, ax=ax, shrink=shrink,
                           orientation='horizontal', pad=0.10,
                           ticks=[vmin, vmin/2.0, 0, vmax/2.0, vmax ],
                           format='%.2e' )


    # Add the colorbar for ratio plots.
    # Set the range to 0.0 and 2.0, with 1.0 in the center
    elif ratio:
        cb = fig.colorbar(im, ax=ax, shrink=shrink,
                          orientation='horizontal', pad=0.10,
                          ticks=[0.0, 0.5, 0.9, 1.1, 1.5, 2.0 ],
                          format='%.1f' )

    # Add the colorbar for absolute magnitude plots
    # Use the tick locator to plot relevant tick intervals
    else:
        cb = fig.colorbar(im, ax=ax, shrink=shrink,
                          orientation='horizontal', pad=0.1 )
        cb.locator = ticker.MaxNLocator(nbins=5)
        cb.update_ticks()

    # can also pass cbar_kwargs to dr.plot() to add colorbar
    cb.set_label(unit)

    # Set the colorbar tick length and label size
    cb.ax.tick_params(length=3, labelsize=10)


def make_pdf(ds1, ds2, filename, on_map=True, diff=False,
             title1='DataSet 1', title2='DataSet 2', unit='',
             matchcbar=False):
    '''Plot all variables in two 2D DataSets, and create a pdf.

    ds1 : xarray.DataSet
        shown on the left column

    ds2 : xarray.DataSet
        shown on the right column

    filename : string
        Name of the pdf file

    on_map : Boolean, optional
        If True (default), use plot_layer() to plot
        If False, use plot_zonal() to plot

    diff : Boolean, optional
        Switch to the color scale for difference plot

    title1, title2 : string, optional
        Title for each DataSet

    unit : string, optional
        Unit shown near the colorbar

    shrink : float, optional
        Horizontal extent of the colorbar (1.0 = width of plot)

    '''

    if on_map:
        plot_func = plot_layer
        subplot_kw = {'projection': ccrs.PlateCarree()}
    else:
        plot_func = plot_zonal
        subplot_kw = None

    # get a list of all variable names in ds1
    # assume ds2 also has those variables
    varname_list = list(ds1.data_vars.keys())

    n_var = len(varname_list)
    print('Benchmarking {} variables'.format(n_var))

    n_row = 3  # how many rows per page. TODO: should this be input argument?
    n_page = (n_var-1) // n_row + 1  # how many pages

    print('generating a {}-page pdf'.format(n_page))
    print('Page: ', end='')

    pdf = PdfPages(filename)

    for ipage in range(n_page):
        print(ipage, end=' ')
        fig, axes = plt.subplots(n_row, 2, figsize=[16, 16],
                                 subplot_kw=subplot_kw)

        # a list of 3 (n_row) variables names
        sub_varname_list = varname_list[n_row*ipage:n_row*(ipage+1)]

        for i, varname in enumerate(sub_varname_list):

            # Get min/max for both datasets to have same colorbar (ewl)
            unitmatch = False
            if matchcbar:
                vmin = min([ds1[varname].data.min(), ds2[varname].data.min()])
                vmax = max([ds1[varname].data.max(), ds2[varname].data.max()])
                unitmatch = ds1[varname].units == ds2[varname].units

            for j, ds in enumerate([ds1, ds2]):
                if on_map:
                    if matchcbar and unitmatch:
                        plot_func(ds[varname], axes[i][j], 
                                  unit=ds[varname].units, diff=diff,
                                  vmin=vmin, vmax=vmax)
                    else:
                        plot_func(ds[varname], axes[i][j], 
                                  unit=ds[varname].units, diff=diff)
                else:
                    # For now, assume zonal mean if plotting zonal (ewl)
                    if matchcbar and unitmatch:
                        plot_func(ds[varname].mean(axis=2), axes[i][j], 
                                  unit=ds[varname].units, diff=diff,
                                  vmin=vmin, vmax=vmax)
                    else:
                        plot_func(ds[varname].mean(axis=2), axes[i][j], 
                                  unit=ds[varname].units, diff=diff)

            # TODO: tweak varname, e.g. Trim "TRC_O3" to "O3"
            axes[i][0].set_title(varname+'; '+title1)
            axes[i][1].set_title(varname+'; '+title2)

            # TODO: unit conversion according to data range of each variable,
            # e.g. mol/mol -> ppmv, ppbv, etc...

        pdf.savefig(fig)
        plt.close(fig)  # don't show in notebook!
    pdf.close()  # close it to save the pdf
    print('done!')


def make_pdf_4panel(ds1, ds2, filename, varlist='', on_map=True, 
                    title1='DataSet 1', title2='DataSet 2', 
                    title3='Abs diff',  title4='Ratio' ):
    '''
    Creates a PDF for all common variables in 2 xarray.Dataset objects,
    as well as their absolute difference and ratio.  
    Creates 4 panels per PDF page, one page per variable.

    Parameters:
    -----------

    ds1 : xarray.DataSet
       First Shown on the top left panel

    ds2 : xarray.DataSet
        Shown on the top right column

    filename : string
        Name of the pdf file

    varlist : List, optional
        List of variable names in ds1 and ds2 to plot.
        If omitted, will default to ds1.data_vars.keys().

    on_map : Boolean, optional
        If True (default), use plot_layer() to plot
        If False, use plot_zonal() to plot

    title1, title2 : string, optional
        Title for each DataSet

    title3 : string, optional
        Title for the absolute difference plot

    title4 : string, optional
        Title for the ratio plot
  
    Notes:
    ------
    (1) The PDF page format is 8.5" x 11", landscape orientation.
    (2) Each variable from ds1 and ds2 is placed onto the same colorbar
        range, to facilitate comparison.
    '''

    # Define the plotting subroutine (and options)
    # for either lon-lat or zonal mean plots
    if on_map:
        plot_func  = plot_layer
        subplot_kw = {'projection': ccrs.PlateCarree()}
    else:
        plot_func  = plot_zonal
        subplot_kw = None

    # If varlist is passed, then use that as the list of common variables
    # in Datasets ds1 and ds2..  Otherwise get a list of variable names in
    # ds1, and assume ds2 also has those variables
    if varlist == '':
        varname_list = list(ds1.data_vars.keys())
    else:
        varname_list = varlist

    # Number of variables
    n_var = len(varname_list)

    # 2 rows and columns per page
    n_row = 2
    n_col = 2

    # One page per variable
    n_page = n_var

    # Printout
    print('Generating a {}-page pdf (one page per variable)'.format(n_page))
    print('Page: ', end='')

    # Open the PDF file object
    pdf = PdfPages(filename)

    # Loop over the number of pages
    for ipage in range(n_page):
                        
        # Print page number
        print(ipage, end=' ')

        # Plot one species per page
        varname = varname_list[ipage]

        # Set up the axes for the subplot panels
        # Assume a standard 8.5 x 11 sheet of paper (landscape)
        fig, axes = plt.subplots(n_row, n_col, 
                                 figsize=[11, 8.5],
                                 subplot_kw=subplot_kw)

        # Extract DataArrays from the respective Datasets
        # and save their units attributes for use below
        dr1   = ds1[varname]
        dr2   = ds2[varname]
        attr1 = dr1.attrs
        attr2 = dr2.attrs

        # Force both datasets to have the same vertical coordinates,
        # otherwise we cannot cleanly take the difference dr2 - dr1.
        # This can happen if there are decimal place differences in
        # the vertical coordinate.  Thanks to Jiawei Zhuang for the fix.
        if 'lev' in dr1.dims and 'lev' in dr2.dims:
            dr1['lev'] = dr2['lev']
            
        # Compute zonal mean if on_map is False
        if on_map == False:
            dr1   = dr1.mean(axis=2)
            dr2   = dr2.mean(axis=2)

            # Manually restore the attributes of each DataArray.
            # When we do any kind of operation on a DataAray it seems
            # to wipe out the associated metadata.
            dr1 = dr1.assign_attrs(attr1)
            dr2 = dr2.assign_attrs(attr2)

        # Get the overall min and max of both DataArrays
        # Always plot both panels on the same colorbar
        vmin_overall = min( [ dr1.data.min(), dr2.data.min() ] )
        vmax_overall = max( [ dr1.data.max(), dr2.data.min() ] )

        # Loop over rows
        for i in range(n_row):

            # Loop over columns
            for j in range(n_col):

                # 1st DataArray
                if i==0 and j==0:
                    dr    = dr1
                    diff  = False
                    ratio = False
                    vmin  = vmin_overall
                    vmax  = vmax_overall
                   
                 # 2nd DataArray
                elif i==0 and j == 1:
                    dr    = dr2
                    diff  = False
                    ratio = False
                    vmin  = vmin_overall
                    vmax  = vmax_overall
                     
                # Difference
                elif i==1 and j == 0:
                    dr    = dr2 - dr1
                    dr    = dr.assign_attrs(attr2)
                    diff  = True
                    ratio = False
                    vmax  = np.max(np.abs(dr.values))
                    vmin  = -vmax

                 # Ratio
                else:                   
                    dr    = safe_div( dr2, dr1 )
                    dr    = dr.assign_attrs(attr2)
                    diff  = False
                    ratio = True
                    vmax  = 2.0
                    vmin  = 0.0

                # Create the plot
                plot_func(dr, axes[i][j], unit=dr.units, 
                          diff=diff, ratio=ratio, 
                          vmin=vmin, vmax=vmax)

                # Add top-of-plot title string
                if i==0 and j==0:
                    axes[i][j].set_title(varname+'; '+title1)
                elif i==0 and j==1:
                    axes[i][j].set_title(varname+'; '+title2)
                elif i==1 and j==0:
                    axes[i][j].set_title(title3)
                else:
                    axes[i][j].set_title(title4)


            # TODO: tweak varname, e.g. Trim "TRC_O3" to "O3"

            # TODO: unit conversion according to data range of each variable,
            # e.g. mol/mol -> ppmv, ppbv, etc...

        # Save this page to the PDF file
        pdf.savefig(fig)
        plt.close(fig)

    # Close the PDF and quit
    pdf.close() 
    print('done!')


def get_common_variables( ds0, ds1, diagstr='' ):

    '''
    Given two xarray Dataset objects, this routine will return a list
    of variable names that are common to both.
    
    NOTE: Will ignore 

    Parameters:
    -----------
    ds0 : xarray Dataset
       The first Dataset object to compare

    ds1 : xarray Dataset
       The second Dataset object to compare

    diagstr : string, optional
       Restricts the search to those elements of ds0 and ds1 that have
       a certain name       

    Returns:
    -------
    varlist : List
       A sorted list object

    '''

    #-----------------------------------------------------------------------
    # If diagstr is specified, then only select that diagnostic,
    # Otherwise, select all diagnostics to be compared & plotted,
    #-----------------------------------------------------------------------
    if diagstr != '':
        vars0  = [k for k in ds0.data_vars.keys() if diagstr in k]
        vars1  = [k for k in ds1.data_vars.keys() if diagstr in k]
    else:
        vars0   = [k for k in ds0.data_vars.keys()]
        vars1   = [k for k in ds1.data_vars.keys()]
        diagstr = 'allVars'

    #-----------------------------------------------------------------------
    # Sort the list of requested diagnostic quantities
    # Only take the diagnostics that are present in both files
    # Exclude some common 1-D and 2-D non-index variables from the output
    #-----------------------------------------------------------------------
    exclude = set( [ 'AREA', 'P0', 'hyam', 'hybm', 'hyai', 'hybi' ] )
    varlist = set(vars0).intersection(vars1)
    varlist = sorted(list(varlist.difference(exclude)))

    return varlist
