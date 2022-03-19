"""
    hyperspectral
    -------------
    
    Metadata
    Datastore I/O
    Mineral map calcs
    Transform

"""

import glob, os
import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from PIL import Image 

def datastore_metadata( path_to_images ):
    """Prepare metadata on datastore"""
    image_map_files = []
    for file in os.listdir( path_to_images ):
        if file.endswith( ".png" ):
            image_map_files.append(file)
    df = pd.DataFrame(image_map_files, columns=["filename"])

    identifiers = parse_identifier( df )
    depths = parse_depths( df )
    image_metadata = pd.concat([df, identifiers, depths], axis=1)
    image_metadata = image_metadata\
        .sort_values(by="box-id")\
        .reset_index( drop=True )
    return image_metadata

def parse_identifier(df):
    """Parse identifier string in mineralmap and mask filenames"""
    identifiers = df["filename"]\
        .str.extract("CMM-(.*)@",expand=False)\
        .str.split("_", expand=True)\
        .astype("int")
    identifiers.columns = ["core-id", "box-id"] 
    return identifiers

def parse_depths(df):
    """Parse depth string in mineralmap and mask filenames"""
    depths = df["filename"]\
        .str.extract("@(.*).png",expand=False)\
        .str.split("_", expand=True)\
        .astype("float")
    depths.columns = ["depth-start", "depth-end"] 
    return depths 

def read_images(path_to_images, image_names, mode="all"):
    """Read images into an ndarray array"""

    if mode == "sample":
     n = 5
    elif mode == "all":
        n = len(image_names)
    else:
        raise Exception("Unknown mode selection. Select sample or all.")
    images = [None] * n
    for i in range(0,n):
        image = np.array( 
            Image.open( os.path.join(path_to_images,  image_names[i]) ) 
            )  
        images[i] = np.flipud( image )
    return images

def read_datastore( path_to_images, mode="sample"):
    """Read image datastore"""

    image_metadata = datastore_metadata( path_to_images )
    image_names = image_metadata["filename"]
    images = read_images(path_to_images, image_names, mode=mode)
    return images

def qc_maskdata( masks ):
    for i in range( len(masks) ):
        if len(masks[i].shape) > 2:
            masks[i]=masks[i][:,:,0].astype( "bool" )
    return masks

def mask_image( image ):
    """Mask background pixels of mineral map image"""
    mask = np.all(image == (0,0,0), axis=-1)
    image_masked = np.copy( image )
    image_masked[mask,:]= [255, 255, 255]
    return image_masked 

def identify_box( mask ):
    """Identify core box columns as mask"""
    bg = np.sum(mask==False, axis=0) == mask.shape[0]
    core_column = label( np.invert(bg) )
    return core_column

def segment_box( box, compartment_mask ):
    """Segment core box""" 
    n_column = np.max( np.unique(compartment_mask) )
    n = n_column+1

    array = [None] * n_column
    for i in range(1,n):
            array[i-1] = box[:, compartment_mask==i]
    return array 

def get_core_pixel_counts( mask, core_compartment ):
    """Get core pixel count per row per compartment"""

    compartments = segment_box( mask, core_compartment )

    n_row = compartments[0].shape[0] 
    n_column = len(compartments)
    n = n_column

    column_pixel_counts = np.zeros( (n_row,n_column ) )
    for j in range(0,n):
        column_pixel_counts[:,j] = np.sum( compartments[j], axis=1 )
    column_pixel_counts[column_pixel_counts==0] = np.nan
    return column_pixel_counts

def get_mineral_pixel_counts( image, mask, mineral_dictionary ):
    """Get all mineral pixel counts per row per compartment"""

    # segment core box mask into columns / compartments 
    core_compartments = identify_box( mask )

    # core pixel counts by row/compartment
    column_pixel_counts = get_core_pixel_counts( mask, core_compartments )

    # define tensor dimensions 
    n_row     = mask.shape[0] 
    n_column  = get_compartment_count( core_compartments )
    n_mineral = len( mineral_dictionary )
    n = n_column

    mineral_pixel_counts = np.zeros( (n_row, n_column, n_mineral) )

    #loop over minerals and core compartments 
    for k, (key,value) in enumerate( mineral_dictionary.items() ):
        mineral = np.all(image == value, axis=-1)
        compartments = segment_box( mineral, core_compartments )
        for j in range(0, n):
            mineral_pixel_counts[:,j,k] = np.sum(compartments[j],axis=1)
    return mineral_pixel_counts

def get_mineral_pixel_prc( mineral_pixel_counts, core_pixel_counts ):
    """Get all mineral pixel percentage per row per compartment"""
    mineral_pixel_prc = mineral_pixel_counts / core_pixel_counts[:,:, np.newaxis]
    return mineral_pixel_prc

def tensor_to_matrix(mineral_tensor):
    """Convert mineral tensor to a mineral matrix"""
    mineral_matrix = np.concatenate( mineral_tensor.transpose(1,0,2), axis=0)
    return mineral_matrix  

def get_compartment_count( compartment_mask ):
    """Number of core columns/compartments"""
    n_compartment = np.max( np.unique( compartment_mask ) )
    return n_compartment

def get_core_compartments( image, mask, astype="image" ):
    """Segment core box into array of core columns"""
    core_compartments = identify_box( mask )
    
    if astype == "image":
        array = segment_box( image, core_compartments )
    elif astype == "mask":
        array = segment_box( mask, core_compartments )
    else:
        array = None
    return array 

def compartment_to_column( compartments ):
    """Convert core box compartments to column"""

    nchannel = compartments[0].shape[2]
    nrow     = compartments[0].shape[0]  
    ncol     = np.array( [item.shape[1] for item in compartments] )

    to_pad = max(ncol) - ncol

    padded=[]
    for (item, value) in zip(compartments,to_pad):
        extra_left, extra_right=[
            np.floor(value/2).astype("int"),
            np.ceil(value/2).astype("int")
            ]
        array = np.pad(item,((0,0),(extra_left,extra_right),(0,0)), 
            mode="constant", constant_values=255)
        padded.append(array)
    column = np.concatenate(padded, axis=0)
    return column

def calc_cumulative_prc( mineral_prc_matrix ):
    """Calculate mineral cumulative percent per row. 
       Zero values assigned NaN. 
    """
    mineral_cumulprc = np.cumsum(mineral_prc_matrix, axis=1)
    mineral_cumulprc[mineral_cumulprc==0] = np.nan
    return mineral_cumulprc

def transpose_and_invert(mineral_matrix, mineral_keys, mineral_colormap):
   """Transpose and invert matrix for composition plot"""
   mineral_plot_matrix = np.flipud( mineral_matrix.transpose() )
   mineral_colors = mineral_colormap[::-1]
   mineral_legend = mineral_keys[::-1]
   return mineral_plot_matrix, mineral_colors, mineral_legend

def get_pixel_depth(mineral_matrix):
    """Core box column pixel depth"""
    n_pixel = mineral_matrix.shape[0]
    return n_pixel

def box_fill( ax, pixel_depth, mineral_matrix, mineral_colors):
   """Create filled plot""" 
   for i,mineral in enumerate(mineral_matrix):
       p = ax.fill_betweenx(pixel_depth, mineral, color=mineral_colors[i])

def plot_box_composition( mineral_prc_matrix, mineral_keys, mineral_colormap ):
    """Plot core box mineral composition"""

    mineral_cumulative = calc_cumulative_prc( mineral_prc_matrix )

    n_pixel = get_pixel_depth(mineral_prc_matrix)
    pixel_depth = np.arange(0,n_pixel)

    mineral_plot_matrix, mineral_colors, mineral_legend = transpose_and_invert( mineral_cumulative, 
        mineral_keys, mineral_colormap)

    fig, ax = plt.subplots(1,1, figsize=(4,12))
    box_fill( ax, pixel_depth, mineral_plot_matrix, mineral_colors)
    ax.invert_yaxis()
    ax.legend(mineral_legend,loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xlabel("[%]")
    ax.set_ylabel("pixel depth");


class MineralMap:
    """Import hyperspectral mineral map from datastore
    
    """
    def __init__(self, path_to_mineral, path_to_mask ):
        """constructor"""
        self._active_index = 0 
        self._problem_index = []
        self._core_box_df = []
        self._core_box_column = []
        self.core_composition = []
        self.core_column = []
        self.core_df = []
        self.minerals = []
        self.masks = []
        self.metadata = []
        self.minerals_dictionary = []
        self.minerals_colormap = []
        self.path_to_mineral = path_to_mineral
        self.path_to_mask = path_to_mask
        
    @property
    def features(self):
        return list(self.minerals_dictionary.keys())

    def read_metadata(self):
        """Read metadata"""
        mineralmap_metadata = datastore_metadata( self.path_to_mineral )
        mask_metadata = datastore_metadata( self.path_to_mask )

        mineralmap_metadata = mineralmap_metadata.rename(columns={
            "filename": "mineral_filename",
            })
        mask_metadata = mask_metadata.rename(columns={
            "filename": "mask_filename",
            })

        joining_variables = ["core-id", "box-id", "depth-start", "depth-end"]
        metadata = mineralmap_metadata.merge(mask_metadata,
            how = "inner", 
            left_on = joining_variables,
            right_on = joining_variables)

        this = metadata.pop( "mineral_filename" )
        metadata.insert(metadata.columns.get_loc("mask_filename"), 
            "mineral_filename", this)
        metadata["box-id"] = metadata["box-id"].astype("int64")
        metadata = metadata.set_index("box-id")
        self.metadata = metadata

    def read_legend(self):
        """Read legend containing mineral map color scheme."""
        legend = np.array( 
            Image.open( os.path.join(self.path_to_mineral, ".." ,  "mineral-legend.png") ) 
            )
        points_to_sample =np.array([
            [17,86],
            [17, 112],
            [17,136],
            [17,158],
            [17,184],
            [17,209],
            [17,235],
            ])
        mineral_keys = [
            "illite",
            "illite-smectite",
            "low-reflectance",
            "montmorillonite",
            "other",
            "smectite-kaolinite",
            "smectite-saponite",
            ]
        pixel_values = []
        pixel_values_scaled = [] 
        for row in points_to_sample:
            value = legend[row[1],row[0]][:3]
            pixel_values.append( tuple(value) )
            pixel_values_scaled.append( tuple(value/255) )
        pixel_values
        mineral_categories = dict( zip(mineral_keys, pixel_values) )
        self.minerals_dictionary = mineral_categories
        self.minerals_colormap = pixel_values_scaled

    def read(self, mode="all"):
        """Read datastore"""
        #read/construct metadata
        self.read_metadata()
        #read mineralmap color scheme
        self.read_legend()
        #read minerals
        image_names = self.metadata["mineral_filename"].values
        self.minerals = np.array( read_images(self.path_to_mineral, image_names, mode=mode ), dtype="object" )
        #read masks 
        image_names = self.metadata["mask_filename"].values
        self.masks  = np.array( qc_maskdata( read_images(self.path_to_mask, image_names, mode=mode ) ), dtype="object" )

    def run_feature_extraction(self):
        """TODO Doc"""
        n = len(self.minerals)
        self._problem_index = []
        features = []
        for i in range( n ):
            self._active_index = i
            self.get_metrics()
            if len(self._core_box_df)==0:
                self._problem_index.append(i)
            features.append(self._core_box_df)
        self.core_df = pd.concat( features ).set_index("box-id")

    def run_corecolumn(self):
        """TODO Doc"""
        n = len(self.minerals)
        columns = []
        for i in range( n ):
            self._active_index = i
            self.get_core_box_column()
            columns.append( self._core_box_column  )
        self.core_column = compartment_to_column( columns )

    def run_composition(self):
        """TODO Doc"""
        self.core_composition = self.get_composition()

    def to_csv(self, path_to_data="" ):
        """TODO Doc"""
        self.core_df.to_csv( path_to_data, index=False )

    def get_metrics(self):
        """Calculate mineral map metrics"""
        mineral = mask_image( self.minerals[self._active_index] )
        mask    = self.masks[ self._active_index ]
        #identify core compartments
        core_compartments = identify_box( mask )
        #core pixel counts 
        core_pixel_counts = get_core_pixel_counts( 
            mask, 
            core_compartments )
        #mineral composition 
        try:
            #mineral pixel counts 
            mineral_pixel_counts = get_mineral_pixel_counts( 
                mineral, 
                mask, 
                self.minerals_dictionary)

            # mineral pixel prc
            mineral_pixel_prc = get_mineral_pixel_prc( 
                mineral_pixel_counts, 
                core_pixel_counts )

            # mineral metric curves
            mineral_metrics = tensor_to_matrix( mineral_pixel_prc )
        except:
            df = pd.DataFrame()
        else:
            # create core_box
            df = pd.DataFrame( 
                mineral_metrics, 
                columns=self.minerals_dictionary.keys() )

            core_box         = self.metadata.index[self._active_index]
            core_start_depth = self.metadata["depth-start"].iloc[self._active_index]
            core_end_depth   = self.metadata["depth-end"].iloc[self._active_index]
            n_pixels = df.shape[0]

            #TODO revisit and calculate resolution 
            core_depth = np.linspace(
                core_start_depth, 
                core_end_depth, 
                n_pixels)
            df.insert(0, 
                "core_depth", core_depth)
            df.insert(0, 
                "box-id",
                np.ones((df.shape[0],1), dtype="int64")*core_box)
        self._core_box_df = df

    def get_core_box_column(self):
        """TODO Doc"""
        mineral = mask_image( self.minerals[self._active_index] )
        mask    = self.masks[ self._active_index ]
        compartments = get_core_compartments(mineral, mask)
        core_column = compartment_to_column( compartments )
        self._core_box_column = core_column

    def get_core_bound(self, box_id=None):
        """Get core depth start and end"""
        if box_id is None:
            df = self.core_df
        else:
            df = self.core_df.loc[box_id,:]
        core_start = df.iloc[0, df.columns.get_loc("core_depth")]
        core_end   = df.iloc[df.shape[0]-1, df.columns.get_loc("core_depth")]
        return core_start, core_end 

    def get_composition(self, box_id=None):
        """TODO Doc"""
        if box_id is None:
            mineral_matrix = self.core_df.loc[:,self.features].to_numpy()
        else:
            mineral_matrix = self.core_df.loc[box_id,self.features].to_numpy()
        mineral_matrix = calc_cumulative_prc(mineral_matrix)
        return mineral_matrix

    def plot_core_box(self, box_id, ptype="minerals"):
        """Preview core box image array"""
        images = getattr(self, ptype)
        images = images[box_id]
        n = len(images)
        fig, ax  = plt.subplots(1,n, figsize=(18,12))
        for item,image in zip(ax,images):
            item.imshow(image, interpolation="none")

    def plot_core_column(self, ax=None):
        """Plot core box column"""
        core_start, core_end = self.get_core_bound()
        if ax is None:
            fig, ax  = plt.subplots(1,1, figsize=(18,12))
        ax.imshow(self.core_column, interpolation="none", aspect=.25, extent=[0,50,core_end,core_start])
        ax.set_ylabel("core depth")

    def plot_core_composition(self, ax=None):
        """TODO Doc"""
        if ax is None:
            fig, ax = plt.subplots(1,1, figsize=(4,12))
        self._plot_composition_helper( ax)

    def plot_core_features(self, minimap=False, comp=False):
        """TODO Doc"""
        n_element = len(self.features)

        if minimap and comp:
            fig, ax = plt.subplots(1,n_element+2, figsize=(18,12), sharey=True)  
            ax_vector = ax[2:]
            index = 1
        elif minimap or comp:
            fig, ax = plt.subplots(1,n_element+1, figsize=(18,12), sharey=True)  
            ax_vector = ax[1:]
            index = 0
        else:
            fig, ax = plt.subplots(1,n_element, figsize=(18,12), sharey=True)
            ax_vector = ax

        invert_axis = True
        if minimap:
            invert_axis = False

        if minimap == True:
            self.plot_core_column( ax=ax[0] )
        if comp == True:
            self.plot_core_composition( ax=ax[index] )
            ax[index].legend(self.features[::-1],loc='center left', bbox_to_anchor=(0, -0.15))
        self._plot_core_helper( ax_vector )
        if invert_axis:
            ax[0].invert_yaxis()
        ax[0].set_ylabel( "core depth" )
        return ax

    def plot_core_box_composition(self, box_id, ax=None):
        """TODO Doc"""
        mineral_matrix = self.get_composition( box_id )
        mineral_stack, mcolors, mlegend = transpose_and_invert(
            mineral_matrix, 
            self.features, 
            self.minerals_colormap)
        #establish core depth vector 
        n_pixel = get_pixel_depth(mineral_matrix)
        core_start, core_end = self.get_core_bound( box_id=box_id)
        core_depth = np.linspace(
                core_start, 
                core_end, 
                n_pixel)
        if ax is None:
            fig, ax = plt.subplots(1,1, figsize=(4,12))
        box_fill( ax, core_depth, mineral_stack, mcolors)
        ax.set_xlabel("[%]")

    def plot_core_box_features( self, box_id, minimap=False, comp=False ):
        """Plot mineral features"""
        n_element = len(self.features)
        
        if minimap and comp:
            fig, ax = plt.subplots(1,n_element+2, figsize=(18,12), sharey=True)  
            ax_vector = ax[2:]
            index = 1
        elif minimap or comp:
            fig, ax = plt.subplots(1,n_element+1, figsize=(18,12), sharey=True)  
            ax_vector = ax[1:]
            index = 0
        else:
            fig, ax = plt.subplots(1,n_element, figsize=(18,12), sharey=True)
            ax_vector = ax

        invert_axis = True
        if minimap:
            invert_axis = False

        if minimap==True:
            core_start, core_end = self.get_core_bound( box_id=box_id)
            n = len(box_id)
            columns = []
            for i in range( n ):
                self._active_index = box_id[i]-1
                self.get_core_box_column()
                columns.append( self._core_box_column  )
                core_column = compartment_to_column( columns )
            ax[0].imshow(core_column, interpolation="none", aspect=50, extent=[0,50,core_end,core_start])
        if comp==True:
            self.plot_core_box_composition(box_id = box_id, ax=ax[index])
            ax[index].legend(self.features[::-1],loc='center left', bbox_to_anchor=(0, -0.15))
        self._plot_core_helper( ax_vector, box_id = box_id )
        if invert_axis:
            ax[0].invert_yaxis()
        ax[0].set_ylabel( "core depth" )
        return ax

    def _plot_core_helper(self, ax, box_id=None):
        """TODO Doc"""
        if box_id is None:
            minerals = self.core_df.loc[:,self.features]
            core_depth = self.core_df["core_depth"]
        else:
            minerals = self.core_df.loc[box_id,self.features]
            core_depth = self.core_df.loc[box_id,"core_depth"]
        mineral_keys = minerals.columns.values
        for i in range( len(ax) ):
                mineral = minerals.loc[:,mineral_keys[i]]
                ax[i].plot(mineral, core_depth)
                ax[i].set_title(mineral_keys[i])

    def _plot_composition_helper(self, ax, box_id=None):
        """TODO Doc"""
        if box_id is None:
            if len(self.core_composition) == 0:
                self.calculate_composition()
            mineral_matrix = self.core_composition
        else:
            mineral_matrix = self.get_composition(box_id = box_id)
        mineral_stack, mcolors, mlegend = transpose_and_invert(
            mineral_matrix, 
            self.features, 
            self.minerals_colormap)
        #establish core depth vector 
        n_pixel = get_pixel_depth(mineral_matrix)
        core_start, core_end = self.get_core_bound( box_id=box_id )
        core_depth = np.linspace(
                core_start, 
                core_end, 
                n_pixel)
        box_fill( ax, core_depth, mineral_stack, mcolors)
        ax.set_xlabel("[%]")

    @classmethod
    def read_ds( cls, path_to_mineral, path_to_mask, mode="all" ):
        """Read hyperspectral datastore"""
        ds = cls(path_to_mineral, path_to_mask)
        ds.read(mode=mode)
        return ds
