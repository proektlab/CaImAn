#!/usr/bin/env python

"""
Utility functions for Neurolabware Scanbox files (.sbx)
"""

import logging
import numpy as np
from numpy import fft
import os
import scipy
import tifffile
from typing import Iterable, Union, Optional

DimSubindices = Union[Iterable[int], slice]
FileSubindices = Union[DimSubindices, Iterable[DimSubindices]]  # can have inds for just frames or also for y, x, z
ChainSubindices = Union[FileSubindices, Iterable[FileSubindices]]  # one to apply to each file, or separate for each file

def loadmat_sbx(filename: str) -> dict:
    """
    this wrapper should be called instead of directly calling spio.loadmat

    It solves the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to fix all entries
    which are still mat-objects
    """
    data_ = scipy.io.loadmat(filename, struct_as_record=False, squeeze_me=True)
    _check_keys(data_)
    return data_


def _check_keys(checkdict: dict) -> None:
    """
    checks if entries in dictionary are mat-objects. If yes todict is called to change them to nested dictionaries.
    Modifies its parameter in-place.
    """

    for key in checkdict:
        if isinstance(checkdict[key], scipy.io.matlab.mio5_params.mat_struct):
            checkdict[key] = _todict(checkdict[key])


def _todict(matobj) -> dict:
    """
    A recursive function which constructs from matobjects nested dictionaries
    """

    ret = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, scipy.io.matlab.mio5_params.mat_struct):
            ret[strg] = _todict(elem)
        else:
            ret[strg] = elem
    return ret


def sbxread(filename: str, subindices: Optional[FileSubindices] = slice(None), channel: Optional[int] = None,
            plane: Optional[int] = None, to32: Optional[bool] = None, auto_process_bidi=True,
            odd_row_ndead: Optional[int] = 0, odd_row_offset: Optional[int] = 0, dead_pix_nan: Optional[Union[str, bool]] = None) -> np.ndarray:
    """
    Load frames of an .sbx file into a new NumPy array

    Args:
        filename: str
            filename should be full path excluding .sbx

        subindices: slice | array-like | tuple[slice | array-like, ...]
            which frames to read (defaults to all)
            if a tuple of non-scalars, specifies slices of up to 4 dimensions in the order (frame, Y, X, Z).

        channel: int | None
            which channel to load (required if data has >1 channel)

        plane: int | None
            set to an int to load only the given plane (converts from 3D to 2D data)
            in the case that len(subindices) == 4, subindices are applied first, then the plane is selected.
        
        to32: bool | None
            whether to read in float32 format (default is to keep as uint16)
            if to32 is None, will be set to True only if necessary to contain nans according to other settings.
            in this case, setting dead_pix_nan to None is the same as setting it to True (will force using NaNs).

        auto_process_bidi: bool
            if true, will automatically estimate dead pixels/offset for bidirectional recordings,
            while assuming they are 0 for unidirectional recordings.
        
        odd_row_ndead, odd_row_offset, dead_pix_nan: see _sbxread_helper.
            odd_row_ndead and odd_row_offset are ignored if auto_process_bidi is True.
    """
    if subindices is None:
        subindices = slice(None)
    
    basename, ext = os.path.splitext(filename)
    if ext == '.sbx':
        filename = basename

    if auto_process_bidi:
        # determine whether recording is bidirectional
        info = loadmat_sbx(filename + '.mat')['info']
        if 'scanmode' in info and info['scanmode'] == 0:
            # bidirectional
            odd_row_ndead = None
            odd_row_offset = None
        else:
            odd_row_ndead = 0
            odd_row_offset = 0
    
    if to32 is None:
        to32 = (odd_row_ndead != 0 or odd_row_offset != 0) and dead_pix_nan in [None, True]

    return _sbxread_helper(filename, subindices=subindices, channel=channel, plane=plane, chunk_size=None, to32=to32,
                           odd_row_ndead=odd_row_ndead, odd_row_offset=odd_row_offset, dead_pix_nan=dead_pix_nan)


def sbx_to_tif(filename: str, fileout: Optional[str] = None, subindices: Optional[FileSubindices] = slice(None),
               bigtiff: Optional[bool] = True, imagej: bool = False, to32: Optional[bool] = None,
               channel: Optional[int] = None, plane: Optional[int] = None, chunk_size: int = 1000, auto_process_bidi=True,
               odd_row_ndead: Optional[int] = 0, odd_row_offset: Optional[int] = 0, dead_pix_nan: Optional[Union[str, bool]] = None):
    """
    Convert a single .sbx file to .tif format

    Args:
        filename: str
            filename should be full path excluding .sbx

        fileout: str | None
            filename to save (defaults to `filename` with .sbx replaced with .tif)

        subindices: slice | array-like | tuple[slice | array-like, ...]
            which frames to read (defaults to all)
            if a tuple of non-scalars, specifies slices of up to 4 dimensions in the order (frame, Y, X, Z).

        to32: bool | None
            whether to save in float32 format (default is to keep as uint16)
            if to32 is None, will be set to True only if necessary to contain nans according to other settings.
            in this case, setting dead_pix_nan to None is the same as setting it to True (will force using NaNs).

        channel: int | None
            which channel to save (required if data has >1 channel)

        plane: int | None
            set to an int to save only the given plane (converts from 3D to 2D data)
            in the case that len(subindices) == 4, subindices are applied first, then the plane is selected.

        chunk_size: int | None
            how many frames to load into memory at once (None = load the whole thing)
        
        auto_process_bidi: bool
            if true, will automatically estimate dead pixels/offset for bidirectional recordings,
            while assuming they are 0 for unidirectional recordings.
        
        odd_row_ndead, odd_row_offset, dead_pix_nan: see _sbxread_helper.
            odd_row_ndead and odd_row_offset are ignored if auto_process_bidi is True.
    """
    # Check filenames
    if fileout is None:
        basename, ext = os.path.splitext(filename)
        if ext == '.sbx':
            filename = basename
        fileout = filename + '.tif'

    if subindices is None:
        subindices = slice(None)

    sbx_chain_to_tif([filename], fileout, [subindices], bigtiff=bigtiff, imagej=imagej, to32=to32,
                     channel=channel, plane=plane, chunk_size=chunk_size, auto_process_bidi=auto_process_bidi,
                     odd_row_ndead=odd_row_ndead, odd_row_offset=odd_row_offset, dead_pix_nan=dead_pix_nan)


def sbx_chain_to_tif(filenames: list[str], fileout: str, subindices: Optional[ChainSubindices] = slice(None),
                     bigtiff: Optional[bool] = True, imagej: bool = False, to32: Optional[bool] = None,
                     channel: Optional[int] = None, plane: Optional[int] = None, chunk_size: int = 1000, auto_process_bidi=True,
                     odd_row_ndead: Optional[int] = 0, odd_row_offset: Optional[int] = 0, dead_pix_nan: Optional[Union[str, bool]] = None) -> None:
    """
    Concatenate a list of sbx files into one tif file.
    Args:
        filenames: list[str]
            each filename should be full path excluding .sbx

        fileout: str
            filename to save, including the .tif suffix
        
        subindices:  Iterable[int] | slice | Iterable[Iterable[int] | slice | tuple[Iterable[int] | slice, ...]]
            see subindices for sbx_to_tif
            can specify separate subindices for each file if nested 2 levels deep; 
            X, Y, and Z sizes must match for all files after indexing.

        odd_row_ndead, odd_row_offset, dead_pix_nan: see _sbxread_helper.
            odd_row_ndead and odd_row_offset are ignored if auto_process_bidi is True.

        to32, channel, plane, chunk_size, auto_process_bidi: see sbx_to_tif
    """
    if subindices is None:
        subindices = slice(None)

    # Validate aggressively to avoid failing after waiting to copy a lot of data
    if isinstance(subindices, slice) or np.isscalar(subindices[0]):
        # One set of subindices to repeat for each file
        subindices = [(subindices,) for _ in filenames]

    elif isinstance(subindices[0], slice) or np.isscalar(subindices[0][0]):
        # Interpret this as being an iterable over dimensions to repeat for each file
        subindices = [subindices for _ in filenames]

    elif len(subindices) != len(filenames):
        # Must be a separate subindices for each file; must match number of files
        raise Exception('Length of subindices does not match length of file list')        

    # Check if any files are bidirectional
    basenames, exts = zip(*[os.path.splitext(file) for file in filenames])
    filenames = [bn if ext == '.sbx' else fn for fn, bn, ext in zip(filenames, basenames, exts)]
    infos = [loadmat_sbx(file + '.mat')['info'] for file in filenames]
    is_bidi = ['scanmode' in info and info['scanmode'] == 0 for info in infos]

    if auto_process_bidi:  # see which recs we want to auto-estimate dead pixels for
        odd_row_ndead = [None if bidi else 0 for bidi in is_bidi]
        odd_row_offset = [None if bidi else 0 for bidi in is_bidi]
        if to32 is None:
            to32 = any(is_bidi) and dead_pix_nan in [None, True]
    else:
        odd_row_ndead = [odd_row_ndead] * len(filenames)
        odd_row_offset = [odd_row_offset] * len(filenames)
        if to32 is None:
            to32 = (odd_row_ndead != 0 or odd_row_offset != 0) and dead_pix_nan in [None, True]

    # Get the total size of the file
    all_shapes = [sbx_shape(file) for file in filenames]
    all_shapes_out = np.stack([_get_output_shape(file, subind)[0] for (file, subind) in zip(filenames, subindices)])

    # Check that X, Y, and Z are consistent
    for dimname, shapes in zip(('Y', 'X', 'Z'), all_shapes_out.T[1:]):
        if np.any(np.diff(shapes) != 0):
            raise Exception(f'Given files have inconsistent shapes in the {dimname} dimension')
    
    # Check that all files have the requested channel
    if channel is None:
        if any(shape[0] > 1 for shape in all_shapes):
            raise Exception('At least one file has multiple channels; must specify channel')
        channel = 0
    elif any(shape[0] <= channel for shape in all_shapes):
        raise Exception('Not all files have the requested channel')

    # Allocate empty tif file with the final shape (do this first to ensure any existing file is overwritten)
    common_shape = tuple(map(int, all_shapes_out[0, 1:]))
    all_n_frames_out = list(map(int, all_shapes_out[:, 0]))
    n_frames_out = sum(all_n_frames_out)
    save_shape = (n_frames_out,) + common_shape

    if plane is not None:
        if len(save_shape) < 4:
            raise Exception('Plane cannot be specified for 2D data')
        save_shape = save_shape[:3]

    # Add a '.tif' extension if not already present
    extension = os.path.splitext(fileout)[1].lower()
    if extension not in ['.tif', '.tiff', '.btf']:
        fileout = fileout + '.tif'

    dtype = np.float32 if to32 else np.uint16
    # Make the file first so we can pass in bigtiff and imagej options; otherwise could create using tifffile.memmap directly
    tifffile.imwrite(fileout, data=None, shape=save_shape, bigtiff=bigtiff, imagej=imagej,
                     dtype=dtype, photometric='MINISBLACK', align=tifffile.TIFF.ALLOCATIONGRANULARITY)

    # Now convert each file
    tif_memmap = tifffile.memmap(fileout, series=0)
    offset = 0
    for filename, subind, file_N, this_ndead, this_offset in zip(filenames, subindices, all_n_frames_out, odd_row_ndead, odd_row_offset):
        _sbxread_helper(filename, subindices=subind, channel=channel, out=tif_memmap[offset:offset+file_N], plane=plane,
                        chunk_size=chunk_size, odd_row_ndead=this_ndead, odd_row_offset=this_offset, dead_pix_nan=dead_pix_nan)
        offset += file_N

    del tif_memmap  # important to make sure file is closed (on Windows)


def sbx_shape(filename: str, info: Optional[dict] = None) -> tuple[int, int, int, int, int]:
    """
    Args:
        filename: str
            filename should be full path excluding .sbx

        info: dict | None
            info struct for sbx file (to avoid re-loading)

    Output: tuple (chans, X, Y, Z, frames) representing shape of scanbox data
    """
    basename, ext = os.path.splitext(filename)
    if ext == '.sbx':
        filename = basename    

    # Load info
    if info is None:
        info = loadmat_sbx(filename + '.mat')['info']

    # Image size
    if 'sz' not in info:
        info['sz'] = np.array([512, 796])
    
    # Scan mode (0 indicates bidirectional)
    if 'scanmode' in info and info['scanmode'] == 0:
        info['recordsPerBuffer'] *= 2

    # Fold lines (multiple subframes per scan) - basically means the frames are smaller and
    # there are more of them than is reflected in the info file
    if 'fold_lines' in info and info['fold_lines'] > 0:
        if info['recordsPerBuffer'] % info['fold_lines'] != 0:
            raise Exception('Non-integer folds per frame not supported')
        n_folds = round(info['recordsPerBuffer'] / info['fold_lines'])
        info['recordsPerBuffer'] = info['fold_lines']
        info['sz'][0] = info['fold_lines']
        if 'bytesPerBuffer' in info:
            info['bytesPerBuffer'] /= n_folds
    else:
        n_folds = 1   

    # Defining number of channels/size factor
    if 'chan' in info:
        info['nChan'] = info['chan']['nchan']
        factor = 1  # should not be used
    else:
        if info['channels'] == 1:
            info['nChan'] = 2
            factor = 1
        elif info['channels'] == 2:
            info['nChan'] = 1
            factor = 2
        elif info['channels'] == 3:
            info['nChan'] = 1
            factor = 2

    # Determine number of frames in whole file
    filesize = os.path.getsize(filename + '.sbx')
    if 'scanbox_version' in info:
        if info['scanbox_version'] == 2:
            info['max_idx'] = filesize / info['recordsPerBuffer'] / info['sz'][1] * factor / 4 - 1
        elif info['scanbox_version'] == 3:
            info['max_idx'] = filesize / np.prod(info['sz']) / info['nChan'] / 2 - 1
        else:
            raise Exception('Invalid Scanbox version')
    else:
        info['max_idx'] = filesize / info['bytesPerBuffer'] * factor - 1

    n_frames = info['max_idx'] + 1    # Last frame

    # Determine whether we are looking at a z-stack
    # Only consider optotune z-stacks - knobby schedules have too many possibilities and
    # can't determine whether it was actually armed from the saved info.
    if info['volscan']:
        n_planes = info['otparam'][2]
    else:
        n_planes = 1
    n_frames //= n_planes

    x = (int(info['nChan']), int(info['sz'][1]), int(info['recordsPerBuffer']), int(n_planes), int(n_frames))
    return x


def sbx_meta_data(filename: str):
    """
    Get metadata for an .sbx file
    Thanks to sbxreader for much of this: https://github.com/jcouto/sbxreader
    Field names and values are equivalent to sbxreader as much as possible

    Args:
        filename: str
            filename should be full path excluding .sbx
    """
    basename, ext = os.path.splitext(filename)
    if ext == '.sbx':
        filename = basename
    
    info = loadmat_sbx(filename + '.mat')['info']

    meta_data = dict()
    n_chan, n_x, n_y, n_planes, n_frames = sbx_shape(filename, info)
    
    # Frame rate
    # use uncorrected recordsPerBuffer here b/c we want the actual # of resonant scans per frame
    meta_data['frame_rate'] = info['resfreq'] / info['recordsPerBuffer'] / n_planes

    # Spatial resolution
    magidx = info['config']['magnification'] - 1
    if 'dycal' in info and 'dxcal' in info:
        meta_data['um_per_pixel_x'] = info['dxcal']
        meta_data['um_per_pixel_y'] = info['dycal']
    else:
        try:
            meta_data['um_per_pixel_x'] = info['calibration'][magidx]['x']
            meta_data['um_per_pixel_y'] = info['calibration'][magidx]['y']
        except (KeyError, TypeError):
            pass
    
    # Optotune depths
    if n_planes == 1:
        meta_data['etl_pos'] = []
    else:
        if 'otwave' in info and not isinstance(info['otwave'], int) and len(info['otwave']):
            meta_data['etl_pos'] = [a for a in info['otwave']]
        
        if 'etl_table' in info:
            meta_data['etl_pos'] = [a[0] for a in info['etl_table']]

    meta_data['scanning_mode'] = 'bidirectional' if info['scanmode'] == 0 else 'unidirectional'
    meta_data['num_frames'] = n_frames
    meta_data['num_channels'] = n_chan
    meta_data['num_planes'] = n_planes
    meta_data['frame_size'] = info['sz']
    meta_data['num_target_frames'] = info['config']['frames']
    meta_data['num_stored_frames'] = info['max_idx'] + 1
    meta_data['stage_pos'] = [info['config']['knobby']['pos']['x'],
                              info['config']['knobby']['pos']['y'],
                              info['config']['knobby']['pos']['z']]
    meta_data['stage_angle'] = info['config']['knobby']['pos']['a']
    meta_data['filename'] = os.path.basename(filename + '.sbx')
    meta_data['resonant_freq'] = info['resfreq']
    meta_data['scanbox_version'] = info['scanbox_version']
    meta_data['records_per_buffer'] = info['recordsPerBuffer']
    meta_data['magnification'] = float(info['config']['magnification_list'][magidx])
    meta_data['objective'] = info['objective']

    for i in range(4):
        if f'pmt{i}_gain' in info['config']:
            meta_data[f'pmt{i}_gain'] = info['config'][f'pmt{i}_gain']

    possible_fields = ['messages', 'event_id', 'usernotes', 'ballmotion',
                       ('frame', 'event_frame'), ('line', 'event_line')]
    
    for field in possible_fields:
        if isinstance(field, tuple):
            field, fieldout = field
        else:
            fieldout = field

        if field in info:
            meta_data[fieldout] = info[field]

    return meta_data


def _sbxread_helper(filename: str, subindices: FileSubindices = slice(None), channel: Optional[int] = None,
                    plane: Optional[int] = None, out: Optional[np.memmap] = None, to32: bool = False, chunk_size: Optional[int] = 1000,
                    odd_row_ndead: Optional[int] = 0, odd_row_offset: Optional[int] = 0, dead_pix_nan: Optional[Union[str, bool]] = None) -> np.ndarray:
    """
    Load frames of an .sbx file into a new NumPy array, or into the given memory-mapped file.

    Args:
        filename: str
            filename should be full path excluding .sbx

        subindices: slice | array-like
            which frames to read (defaults to all)

        channel: int | None
            which channel to save (required if data has >1 channel)

        out: np.memmap | None
            existing memory-mapped file to write into
        
        to32: bool
            whether to convert to float32 if creating a new array. ignored if writing into an existing array.

        plane: int | None
            set to an int to load only the given plane (converts from 3D to 2D data)
            in the case that len(subindices) == 4, subindices are applied first, then the plane is selected.

        chunk_size: int | None
            how many frames to load into memory at once (None = load the whole thing)

        odd_row_ndead: int | None
            how many columns at the left of the odd rows (i.e., starting with the 2nd row) to remove and
            replace according to dead_pix_mode. these pixels are unrecorded in bidirectional mode and end up "saturated."
            this applies to the image before any cropping by subindices.
            defaults to 0, but if None, tries to estimate automatically.
        
        odd_row_offset: int
            how many pixels rows 1,3,5... are offset relative to rows 0,2,4... (positive = right).
            this can occur due to misalignment during bidirectional scanning. rows are shifted to
            correct the offset, including moving pixels in the rightmost columns up or down as necessary.
            defaults to 0, but if None, tries to estimate automatically.
        
        dead_pix_nan: str | bool | None
            how to replace dead pixels identified by odd_row_nsaturated and odd_row_offset. this can be any
            valid value for params.motion['border_nan'] except 'min'. Default is 'copy' if writing to int-valued array, else True (= NaN).
            Note that NaNs are replaced by interpolated values if params.preprocess['check_nan'] is true.
    """ 
    basename, ext = os.path.splitext(filename)
    if ext == '.sbx':
        filename = basename

    # Normalize so subindices is a list over dimensions
    if isinstance(subindices, slice) or np.isscalar(subindices[0]):
        subindices = [subindices]
    else:
        subindices = list(subindices)

    # Load info
    info = loadmat_sbx(filename + '.mat')['info']

    # Get shape (and update info)
    data_shape = sbx_shape(filename, info)  # (chans, X, Y, Z, frames)
    n_chans, n_x, n_y, n_planes, n_frames = data_shape
    is3D = n_planes > 1

    # Fill in missing dimensions in subindices
    subindices += [slice(None) for _ in range(max(0, 3 + is3D - len(subindices)))]

    if channel is None:
        if n_chans > 1:
            raise Exception('Channel input required for multi-chanel data')
        channel = 0
    elif channel >= n_chans:
        raise Exception(f'Channel input out of range (data has {n_chans} channels)')

    if 'scanbox_version' in info and info['scanbox_version'] == 3:
        frame_size = round(np.prod(info['sz']) * info['nChan'] * 2 * n_planes)
    else:
        frame_size = round(info['sz'][1] * info['recordsPerBuffer'] * 2 * info['nChan'] * n_planes)
    if frame_size <= 0:
        raise Exception('Invalid scanbox metadata')

    save_shape, subindices = _get_output_shape(data_shape, subindices)
    n_frames_out = save_shape[0]
    if plane is not None:
        if len(save_shape) < 4:
            raise Exception('Plane cannot be specified for 2D data')
        save_shape = save_shape[:3]

    if out is not None and out.shape != save_shape:
        raise Exception('Existing memmap is the wrong shape to hold loaded data')

    # Read from .sbx file, using memmap to avoid loading until necessary
    sbx_mmap = np.memmap(filename + '.sbx', mode='r', dtype='uint16', shape=data_shape, order='F')
    sbx_mmap = np.transpose(sbx_mmap, (0, 4, 2, 1, 3))  # to (chans, frames, Y, X, Z)
    sbx_mmap = sbx_mmap[channel]
    if not is3D:  # squeeze out singleton plane dim
        sbx_mmap = sbx_mmap[..., 0]
    elif plane is not None:  # select plane relative to subindices
        sbx_mmap = sbx_mmap[..., subindices[-1][plane]]
        subindices = subindices[:-1]

    # estimate dead pixels if necessary
    if odd_row_ndead is None:
        odd_row_ndead = _estimate_odd_row_nsaturated(sbx_mmap[0])
        if odd_row_ndead == 0:
            logging.info('Found no dead pixels at left of odd rows')

    if odd_row_ndead > 0:
        logging.info(f'Correcting {odd_row_ndead} dead pixels at left of odd rows')

    # estimate row offset if necessary
    if odd_row_offset is None:
        n_samps = min(len(sbx_mmap), 300)
        sample = sbx_mmap[np.linspace(0, n_samps, endpoint=False, dtype=int)]
        odd_row_offset = _estimate_odd_row_offset(sample)
        if odd_row_offset == 0:
            logging.info(f'Found no line phase offset')

    if odd_row_offset != 0:
        logging.info(f'Correcting line phase offset of {odd_row_offset}')
    
    if (out is None and not to32) or (out is not None and out.dtype.kind != 'f'):
        # saving to int array, default to copy
        if dead_pix_nan is None:
            dead_pix_nan = 'copy'
        elif dead_pix_nan == True:
            raise Exception('Cannot write NaN values to int array; dead_pix_nan cannot be True')
    elif dead_pix_nan is None:  # saving to float array, default to NaNs
        dead_pix_nan = True

    if odd_row_ndead == 0 and odd_row_offset == 0:
        # this list specifies how to index the input and output arrays when copying data.
        # format of each entry: (<tuple of out-indices>, <tuple of in-indices>)
        # or: (<tuple of out-indices>, constant)
        # each entry is applied in order.
        inds_sets = [((range(n_frames_out),), np.ix_(*subindices))]
    else:
        # load even and odd rows separately to implement shift and correct dead pixels
        inds_sets = _make_inds_sets_with_corrections(n_frames, n_y, n_x, subindices, save_shape,
                                                     odd_row_ndead, odd_row_offset, dead_pix_nan)

    if chunk_size is None:
        # load a contiguous block all at once
        chunk_size = n_frames_out

    n_remaining = n_frames_out
    offset = 0
    while n_remaining > 0:
        this_chunk_size = min(n_remaining, chunk_size)

        for out_inds, in_inds in inds_sets:
            if np.isscalar(in_inds):
                chunk = in_inds
            else:
                # Note: important to copy the data here instead of making a view,
                # so the memmap can be closed (achieved by advanced indexing)
                chunk = sbx_mmap[(in_inds[0][offset:offset+this_chunk_size],) + in_inds[1:]]
                # Note: SBX files store the values strangely, it's necessary to invert each uint16 value to get the correct ones
                np.invert(chunk, out=chunk)  # avoid copying, may be large

            if out is None and not to32 and out_inds == (range(n_frames_out),) and this_chunk_size == n_frames_out:
                # avoid copying again when just loading all data
                out = chunk
            else:
                if out is None:
                    out = np.empty(save_shape, dtype=(np.float32 if to32 else np.uint16))
                out[(out_inds[0][offset:offset+this_chunk_size],) + out_inds[1:]] = chunk

        n_remaining -= this_chunk_size
        offset += this_chunk_size

    del sbx_mmap  # Important to close file (on Windows)

    if isinstance(out, np.memmap):
        out.flush()    
    return out


def _interpret_subindices(subindices: DimSubindices, dim_extent: int) -> tuple[Iterable[int], int]:
    """
    Given the extent of a dimension in the corresponding recording, obtain an iterable over subindices 
    and the step size (or 0 if the step size is not uniform).
    """
    if isinstance(subindices, slice):
        iterable_elements = range(dim_extent)[subindices]
        skip = iterable_elements.step

        if subindices.stop is not None and np.isfinite(subindices.stop) and subindices.stop > dim_extent:
            logging.warning(f'Only {dim_extent} frames or pixels available to load ' +
                            f'(requested up to {subindices.stop})')
    else:
        iterable_elements = subindices
        if isinstance(subindices, range):
            skip = subindices.step
        else:
            skip = 0

    return iterable_elements, skip


def _get_output_shape(filename_or_shape: Union[str, tuple[int, ...]], subindices: FileSubindices
                      ) -> tuple[tuple[int, ...], FileSubindices]:
    """
    Helper to determine what shape will be loaded/saved given subindices
    Also returns back the subindices with slices transformed to ranges, for convenience
    """
    if isinstance(subindices, slice) or np.isscalar(subindices[0]):
        subindices = (subindices,)
    
    n_inds = len(subindices)  # number of dimensions that are indexed

    if isinstance(filename_or_shape, str):
        data_shape = sbx_shape(filename_or_shape)
    else:
        data_shape = filename_or_shape
    
    n_x, n_y, n_planes, n_frames = data_shape[1:]
    is3D = n_planes > 1
    if n_inds > 3 + is3D:
        raise Exception('Too many dimensions in subdindices')
    
    shape_out = [n_frames, n_y, n_x, n_planes] if is3D else [n_frames, n_y, n_x]
    subinds_out = []
    for i, (dim, subind) in enumerate(zip(shape_out, subindices)):
        iterable_elements = _interpret_subindices(subind, dim)[0]
        shape_out[i] = len(iterable_elements)
        subinds_out.append(iterable_elements)

    return tuple(shape_out), tuple(subinds_out)


def _estimate_odd_row_nsaturated(frame: np.memmap) -> int:
    """
    Based on a single frame, estimate how many columns on the left are "dead"
    (i.e., every other line is saturated) in odd rows.

    Args:
        frame: memmmap of shape (Y, X, [Z])
            One frame of data, which is still in inverted format

    Returns:
        ndead: int
            Number of dead columns on the left of the frame in odd rows
    """
    mean_axes = 0 if frame.ndim == 2 else (0, 2)
   
    col_profile = np.mean(frame[1::2], axis=mean_axes)
    not_dead = np.flatnonzero(col_profile > 0)
    if len(not_dead) == 0:
        logging.warning(f'Odd rows are saturated in all columns!')
        return frame.shape[1]
    else:
        return not_dead[0]


def _estimate_odd_row_offset(frames: np.ndarray) -> int:
    """
    Returns the bidirectional phase offset, the offset between lines that sometimes occurs in line scanning.
    Copied from suite2p (GPL)
    Copright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
    """
    Lx = frames.shape[2]

    # compute phase-correlation between lines in x-direction
    d1 = fft.fft(frames[:, 1::2, :], axis=2)
    d1 /= np.abs(d1) + 1e-5

    d2 = np.conj(fft.fft(frames[:, ::2, :], axis=2))
    d2 /= np.abs(d2) + 1e-5
    d2 = d2[:, :d1.shape[1], :]

    cc = np.real(fft.ifft(d1 * d2, axis=2))
    cc = cc.mean(axis=tuple(range(2)) + tuple(range(3, cc.ndim)))
    cc = fft.fftshift(cc)

    bidiphase = np.argmax(cc[-10 + Lx // 2:11 + Lx // 2]) - 10
    return bidiphase


IndsList = tuple[np.ndarray, ...]   # (each element an output from np.ix_)
def _make_inds_sets_with_corrections(n_frames: int, n_y: int, n_x: int, subindices: tuple[Iterable[int], ...],
                                     save_shape: tuple[int, ...], odd_row_ndead: int, odd_row_offset: int,
                                     dead_pix_nan: Union[str, bool]) -> list[tuple[IndsList, Union[int, float, IndsList]]]:
    """Compute indices/constant values for reading and writing data given dead pixels and/or bidi offset"""
    out_inds_t = np.arange(save_shape[0])
    out_inds_y = np.arange(save_shape[1])
    out_inds_x = np.arange(save_shape[2])
    in_inds_t = np.array(subindices[0]) if len(subindices) > 0 else np.arange(n_frames)
    in_inds_y = np.array(subindices[1]) if len(subindices) > 1 else np.arange(n_y)
    in_inds_x = np.array(subindices[2]) if len(subindices) > 2 else np.arange(n_x)

    b_even_row = in_inds_y % 2 == 0
    inds_sets = []

    # when odd_row_offset is odd, shift right 1 pixel more than alternate rows are shifted left
    # (so 1 pixel gets "eaten")
    lshift = abs(odd_row_offset) // 2
    rshift = abs(odd_row_offset) - lshift

    if odd_row_offset >= 0:
        even_shift = -rshift
        odd_shift = lshift
    else:
        even_shift = lshift
        odd_shift = -rshift

    e2e_mask = (0 <= in_inds_x + even_shift) & (in_inds_x + even_shift < n_x)
    inds_sets.append((
        np.ix_(out_inds_t, out_inds_y[b_even_row], out_inds_x[e2e_mask]),
        np.ix_(in_inds_t, in_inds_y[b_even_row], in_inds_x[e2e_mask] + even_shift, *subindices[3:])
    ))

    o2o_mask = (odd_row_ndead <= in_inds_x + odd_shift) & (in_inds_x + odd_shift < n_x)
    inds_sets.append((
        np.ix_(out_inds_t, out_inds_y[~b_even_row], out_inds_x[o2o_mask]),
        np.ix_(in_inds_t, in_inds_y[~b_even_row], in_inds_x[o2o_mask] + odd_shift, *subindices[3:])
    ))

    # wrapping pixels at ends of rows
    wrap_mask = in_inds_x + lshift >= n_x
    to_even = odd_row_offset < 0
    wrap_out_y = out_inds_y[b_even_row == to_even]
    wrap_in_y = in_inds_y[b_even_row == to_even] + (1 if to_even else -1)
    if odd_row_offset != 0:
        inds_sets.append((
            np.ix_(out_inds_t, wrap_out_y, out_inds_x[wrap_mask]),
            np.ix_(in_inds_t, wrap_in_y, -(in_inds_x[wrap_mask] + lshift - n_x + 1), *subindices[3:])
        ))

    even_dead_mask = in_inds_x + even_shift < 0
    odd_dead_mask = in_inds_x + odd_shift < odd_row_ndead 
    
    if dead_pix_nan == 'copy':
        inds_sets.append((
            np.ix_(out_inds_t, out_inds_y[b_even_row], out_inds_x[even_dead_mask]),
            np.ix_(in_inds_t, in_inds_y[b_even_row], [0], *subindices[3:])
        ))

        inds_sets.append((
            np.ix_(out_inds_t, out_inds_y[~b_even_row], out_inds_x[odd_dead_mask]),
            np.ix_(in_inds_t, out_inds_y[~b_even_row], [odd_row_ndead], *subindices[3:])
        ))
    else:
        inds_sets.append((
            np.ix_(out_inds_t, out_inds_y[b_even_row], out_inds_x[even_dead_mask]),
            np.nan if dead_pix_nan else 0
        ))

        inds_sets.append((
            np.ix_(out_inds_t, out_inds_y[~b_even_row], out_inds_x[odd_dead_mask]),
            np.nan if dead_pix_nan else 0
        ))
    
    return inds_sets
    