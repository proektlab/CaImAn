#!/usr/bin/env python

"""
Utility functions for Neurolabware Scanbox files (.sbx)
"""
from ipyparallel import AsyncResult
import logging
import numpy as np
from numpy import fft
import os
import scipy
from scipy import ndimage, interpolate
import tifffile
from tqdm import tqdm, trange
from typing import Union, Optional, Sequence, cast, Any

from trycast import isassignable

DimSubindices = Union[Sequence[int], slice]
FileSubindices = Union[DimSubindices, Sequence[DimSubindices]]  # can have inds for just frames or also for y, x, z
ChainSubindices = Union[FileSubindices, Sequence[FileSubindices]]  # one to apply to each file, or separate for each file

def loadmat_sbx(filename: str) -> dict[str, Any]:
    """
    this wrapper should be called instead of directly calling spio.loadmat

    It solves the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to fix all entries
    which are still mat-objects
    """
    data_ = scipy.io.loadmat(filename, struct_as_record=False, squeeze_me=True)
    _check_keys(data_)
    return data_['info']


def _check_keys(checkdict: dict) -> None:
    """
    checks if entries in dictionary are mat-objects. If yes todict is called to change them to nested dictionaries.
    Modifies its parameter in-place.
    """

    for key in checkdict:
        if isinstance(checkdict[key], scipy.io.matlab.mat_struct):
            checkdict[key] = _todict(checkdict[key])


def _todict(matobj) -> dict:
    """
    A recursive function which constructs from matobjects nested dictionaries
    """

    ret = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, scipy.io.matlab.mat_struct):
            ret[strg] = _todict(elem)
        else:
            ret[strg] = elem
    return ret


def sbxread(filename: str, subindices: Optional[FileSubindices] = slice(None), channel: Optional[int] = None,
            plane: Optional[int] = None, to32: Optional[bool] = None, odd_row_ndead: Optional[int] = None,
            odd_row_offset: Optional[int] = 0, force_estim_ndead_offset: bool = False, interp: bool = True,
            dead_pix_mode: Union[str, bool] = 'copy', dview=None, quiet=False) -> np.ndarray:
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

        force_estim_ndead_offset: bool
            when this flag is false (default) and None is passed to odd_row_ndead and/or odd_row_offset, the corrections are
            estimated automatically for bidirectional recordings and set to 0 for unidirectional recordings.
            if this flag is true, the behavior is changed to automatically estimate ndead and offset for unidirectional recordings as well.
        
        odd_row_ndead, odd_row_offset, interp, dead_pix_mode: see _sbxread_helper.
    """
    if subindices is None:
        subindices = slice(None)
    
    basename, ext = os.path.splitext(filename)
    if ext == '.sbx':
        filename = basename

    if not force_estim_ndead_offset and (odd_row_ndead is None or odd_row_offset is None):
        # if recording is unidirectional, switch Nones to 0
        info = loadmat_sbx(filename + '.mat')
        if 'scanmode' in info and info['scanmode'] != 0:
            # unidirectional
            if odd_row_ndead is None:
                odd_row_ndead = 0
            if odd_row_offset is None:
                odd_row_offset = 0
    
    if to32 is None:
        to32 = (odd_row_ndead != 0 or odd_row_offset != 0) and dead_pix_mode == True

    return _sbxread_helper(filename, subindices=subindices, channel=channel, plane=plane, chunk_size=None, to32=to32,
                           odd_row_ndead=odd_row_ndead, odd_row_offset=odd_row_offset, interp=interp, dead_pix_mode=dead_pix_mode,
                           dview=dview, quiet=quiet)


def sbx_to_tif(filename: str, fileout: Optional[str] = None, subindices: Optional[FileSubindices] = slice(None),
               bigtiff: Optional[bool] = True, imagej: bool = False, to32: Optional[bool] = None,
               channel: Optional[int] = None, plane: Optional[int] = None, chunk_size: Optional[int] = 100,
               odd_row_ndead: Optional[int] = None, odd_row_offset: Optional[int] = 0, force_estim_ndead_offset: bool = False,
               interp: bool = True, dead_pix_mode: Union[str, bool] = 'copy', dview=None) -> None:
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

        channel: int | None
            which channel to save (required if data has >1 channel)

        plane: int | None
            set to an int to save only the given plane (converts from 3D to 2D data)
            in the case that len(subindices) == 4, subindices are applied first, then the plane is selected.

        chunk_size: int | None
            how many frames to load into memory at once (None = load the whole thing)
        
        force_estim_ndead_offset: bool
            when this flag is false (default) and None is passed to odd_row_ndead and/or odd_row_offset, the corrections are
            estimated automatically for bidirectional recordings and set to 0 for unidirectional recordings.
            if this flag is true, the behavior is changed to automatically estimate ndead and offset for unidirectional recordings as well.
        
        odd_row_ndead, odd_row_offset, interp, dead_pix_mode: see _sbxread_helper.
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
                     channel=channel, plane=plane, chunk_size=chunk_size, force_estim_ndead_offset=force_estim_ndead_offset,
                     odd_row_ndead=odd_row_ndead, odd_row_offset=odd_row_offset, interp=interp,
                     dead_pix_mode=dead_pix_mode, dview=dview)


def broadcast_chain_subindices(maybe_subindices: Optional[ChainSubindices], n_files: int) -> Sequence[FileSubindices]:
    subindices = slice(None) if maybe_subindices is None else maybe_subindices

    # Validate aggressively to avoid failing after waiting to copy a lot of data
    if isassignable(subindices, DimSubindices):
        # subindices over time to repeat for each file
        dim_subindices = cast(DimSubindices, subindices)
        chain_subindices = [(dim_subindices,)] * n_files
    elif isassignable(subindices, FileSubindices):
        # sequence of dimension subindices to repeat for each file
        file_subindices = cast(FileSubindices, subindices)
        chain_subindices = [file_subindices] * n_files
    else:
        # sequence of sequences of dimension subindices
        chain_subindices = cast(Sequence[FileSubindices], subindices)
        if len(chain_subindices) != n_files:
            # Must be a separate subindices for each file; must match number of files
            raise Exception('Length of subindices does not match length of file list')    
    return chain_subindices


def sbx_chain_to_tif(filenames: list[str], fileout: str, subindices: Optional[ChainSubindices] = slice(None),
                     bigtiff: Optional[bool] = True, imagej: bool = False, to32: Optional[bool] = None,
                     channel: Optional[int] = None, plane: Optional[int] = None, chunk_size: Optional[int] = 100,
                     odd_row_ndead: Union[Optional[int], Sequence[Optional[int]]] = None, odd_row_offset: Union[Optional[int], Sequence[Optional[int]]] = 0,
                     force_estim_ndead_offset: bool = False, interp: bool = True, dead_pix_mode: Union[str, bool] = 'copy', dview=None
                     ) -> list[int]:
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

        odd_row_ndead, odd_row_offset, interp, dead_pix_mode: see _sbxread_helper.

        to32, channel, plane, chunk_size, force_estim_ndead_offset: see sbx_to_tif
    Returns:
        all_n_frames_out: list[int]
            number of frames from each file saved to the output
    """
    subindices = broadcast_chain_subindices(subindices, len(filenames))     

    # Check if any files are bidirectional
    basenames, exts = zip(*[os.path.splitext(file) for file in filenames])
    filenames = [bn if ext == '.sbx' else fn for fn, bn, ext in zip(filenames, basenames, exts)]

    if not isinstance(odd_row_ndead, list):
        if isinstance(odd_row_ndead, Sequence):
            odd_row_ndead = list(odd_row_ndead)
        else:
            odd_row_ndead = [odd_row_ndead] * len(filenames)
    
    if not isinstance(odd_row_offset, list):
        if isinstance(odd_row_offset, Sequence):
            odd_row_offset = list(odd_row_offset)
        else:
            odd_row_offset = [odd_row_offset] * len(filenames)

    if not force_estim_ndead_offset:
        # change None to 0 for unidirectional scans
        for i, (file, ndead, offset) in enumerate(zip(filenames, odd_row_ndead, odd_row_offset)):
            info = loadmat_sbx(file + '.mat')
            bidi = 'scanmode' in info and info['scanmode'] == 0
            if not bidi:
                if ndead is None:
                    odd_row_ndead[i] = 0
                if offset is None:
                    odd_row_offset[i] = 0

    might_do_correction = (
        any(ndead != 0 for ndead in odd_row_ndead) or
        any(offset != 0 for offset in odd_row_offset))
    if to32 is None:
        # if we will be adding nans to the final image, must convert to float32
        to32 = dead_pix_mode == True and might_do_correction

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

    # Make the output file and prepare for parallel processing
    dtype = np.dtype(np.float32 if to32 else np.uint16)
    frame_slices, out_memmap_args = _prepare_concat_output_memmap(fileout, save_shape, all_n_frames_out, dtype=dtype,
                                                                  bigtiff=bigtiff, imagej=imagej)

    args = ((out_args, filename, subind, channel, plane, False, chunk_size, this_ndead, this_offset, interp, dead_pix_mode)
            for out_args, filename, subind, this_ndead, this_offset in
            zip(out_memmap_args, filenames, subindices, odd_row_ndead, odd_row_offset, strict=True))
    
    pbar = trange(len(filenames), desc='Converting each file...', unit='file')
    if dview is not None and (might_do_correction or len(filenames) > 10):  # (is it worth doing this step in parallel?)
        if 'DirectView' in str(type(dview)):
            # ipyparallel, could be across computers so don't do in place
            tif_dtype = out_memmap_args[0]['dtype']
            tif_data_offset = out_memmap_args[0]['offset']
            tif_memmap = np.memmap(fileout, tif_dtype, 'r+', tif_data_offset, save_shape, 'C')
            args_not_inplace = ((None,) + a[1:] for a in args)
            for res, out_slice, _ in zip(dview.map_async(_sbxread_worker, args_not_inplace), frame_slices, pbar):
                tif_memmap[out_slice] = res
            tif_memmap.flush()
            del tif_memmap  # important to make sure file is closed (on Windows)
        else:
            for res, _ in zip(dview.map(_sbxread_worker, args), pbar):
                pass
    else:
        for arglist, _ in zip(args, pbar, strict=True):
            _sbxread_worker(arglist, dview)
    pbar.close()
    return all_n_frames_out


def sbx_shape(filename: str, info: Optional[dict[str, Any]] = None) -> tuple[int, int, int, int, int]:
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
        info = loadmat_sbx(filename + '.mat')

    # Image size
    if 'sz' not in info:
        info['sz'] = np.array([512, 796])

    # Fold lines (multiple subframes per scan) - basically means the frames are smaller and
    # there are more of them than is reflected in the info file
    if 'fold_lines' in info and info['fold_lines'] > 0:
        if info['sz'][0] % info['fold_lines'] != 0:
            raise Exception('Non-integer folds per frame not supported')

        info['sz'][0] = info['fold_lines']
        if 'bytesPerBuffer' in info:
            n_folds = round(info['sz'][0] / info['fold_lines'])
            info['bytesPerBuffer'] /= n_folds


    # Defining number of channels/size factor
    if 'chan' in info:
        info['nChan'] = info['chan']['nchan']
    elif info['channels'] == 1:
        info['nChan'] = 2
    else:
        info['nChan'] = 1

    # Determine number of frames in whole file
    filesize = os.path.getsize(filename + '.sbx')
    if 'scanbox_version' in info:
        if info['scanbox_version'] in [2, 3]:
            info['max_idx'] = filesize / np.prod(info['sz']) / info['nChan'] / 2 - 1
        else:
            raise Exception('Invalid Scanbox version')
    else:
        info['max_idx'] = filesize / info['bytesPerBuffer'] * (2 // info['nChan']) - 1

    n_frames = info['max_idx'] + 1    # Last frame

    # Determine whether we are looking at a z-stack
    # Only consider optotune z-stacks - knobby schedules have too many possibilities and
    # can't determine whether it was actually armed from the saved info.
    if info['volscan']:
        n_planes = info['otparam'][2]
    else:
        n_planes = 1
    n_frames //= n_planes

    x = (int(info['nChan']), int(info['sz'][1]), int(info['sz'][0]), int(n_planes), int(n_frames))
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
    
    info = loadmat_sbx(filename + '.mat')

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


def get_odd_row_ndead(filename: str) -> int:
    """From sbx file (assumed to be bidirectional), estimate number of dead columns at left of odd rows."""
    info = loadmat_sbx(filename + '.mat')
    data_shape = sbx_shape(filename, info)  # (chans, X, Y, Z, frames)
    sbx_mmap = np.memmap(filename + '.sbx', mode='r', dtype='uint16', shape=data_shape, order='F')
    sbx_mmap = np.transpose(sbx_mmap, (0, 4, 2, 1, 3))  # to (chans, frames, Y, X, Z)
    odd_row_ndead = _estimate_odd_row_nsaturated(sbx_mmap[0, 0])
    return odd_row_ndead


def _prepare_concat_output_memmap(filename: str, full_shape: tuple[int, ...], frames_per_section: list[int], dtype,
                                  bigtiff: Optional[bool] = True, imagej=False) -> tuple[list[slice], list[dict]]:
    """
    Write an empty output tif file with the given full_shape. frames_per_section specifies how many frames are from each subsection
    of the file (for example, different trials that are being concatenated)

    Returns:
    - a list of slices for the 1st dimension of the output file for each subsection
    - a list of dicts of np.memmap arguments to use to memmap each subsection of the file
    """
    if sum(frames_per_section) != full_shape[0]:
        raise ValueError('Frames per section is not a valid division of the total frames')

    # write the file
    res = tifffile.imwrite(filename, data=None, shape=full_shape, returnoffset=True, bigtiff=bigtiff, imagej=imagej,
                           dtype=dtype, photometric='MINISBLACK', byteorder='<', align=tifffile.TIFF.ALLOCATIONGRANULARITY)
    if res is None:
        raise RuntimeError('Tiff file is not memmappable')
    tif_data_offset, _ = res 
    tif_dtype = '<' + dtype.char

    # make arguments for each sub-section
    offsets = np.insert(np.cumsum(frames_per_section), 0, 0)  # frame offset for each file
    frame_slices = [slice(offset0, offset1) for offset0, offset1 in zip(offsets[:-1], offsets[1:])]
    byte_offsets = [int(tif_data_offset + np.prod(full_shape[1:]) * dtype.itemsize * offset)
                    for offset in np.array(offsets[:-1])]
    shapes = [(int(d),) + full_shape[1:] for d in np.diff(offsets)]
    out_memmap_args = [{
        'filename': filename,
        'dtype': tif_dtype,
        'mode': 'r+',
        'offset': offset,
        'shape': shape,
        'order': 'C'}
        for offset, shape in zip(byte_offsets, shapes, strict=True)]
    return frame_slices, out_memmap_args


def _sbxread_helper(filename: str, subindices: FileSubindices = slice(None), channel: Optional[int] = None,
                    plane: Optional[int] = None, out: Optional[np.memmap] = None, to32: bool = False, chunk_size: Optional[int] = 100,
                    odd_row_ndead: Optional[int] = 0, odd_row_offset: Optional[int] = 0,
                    interp=True, dead_pix_mode: Union[str, bool] = 'copy', dview=None, quiet=False) -> np.ndarray:
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
        
        interp: bool
            whether to interpolate dead pixels that fall within the convex hull of known pixels (i.e., from odd_row_offset).
            otherwise, they will be filled according to dead_pix_mode.

        dead_pix_mode: str | bool
            how to replace dead pixels identified by odd_row_nsaturated and odd_row_offset. Same options as params.motion['border_nan'],
            and True (NaN) is invalid if 'to32' is False. if interp is True, this only sets the extrapolation mode.
    """ 
    basename, ext = os.path.splitext(filename)
    if ext == '.sbx':
        filename = basename

    # Normalize so subindices is a list over dimensions
    if isassignable(subindices, DimSubindices):
        dim_subindices = cast(DimSubindices, subindices)
        subindices = [dim_subindices]
    else:
        subindices = list(cast(Sequence[DimSubindices], subindices))

    # Load info
    info = loadmat_sbx(filename + '.mat')

    # Get shape (and update info)
    data_shape = sbx_shape(filename, info)  # (chans, X, Y, Z, frames)
    n_chans, n_x, n_y, n_planes, n_frames = data_shape
    is3D = n_planes > 1

    # Fill in missing dimensions in subindices
    subindices += [slice(None) for _ in range(max(0, 3 + is3D - len(subindices)))]

    if channel is None:
        if n_chans > 1:
            raise Exception('Channel input required for multi-channel data')
        channel = 0
    elif channel >= n_chans:
        raise Exception(f'Channel input out of range (data has {n_chans} channels)')

    if 'scanbox_version' in info and info['scanbox_version'] == 3:
        frame_size = round(np.prod(info['sz']) * info['nChan'] * 2 * n_planes)
    else:
        frame_size = round(info['sz'][1] * info['recordsPerBuffer'] * 2 * info['nChan'] * n_planes)
    if frame_size <= 0:
        raise Exception('Invalid scanbox metadata')

    save_shape, subind_seqs = _get_output_shape(data_shape, subindices)
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
        sbx_mmap = sbx_mmap[..., subind_seqs[-1][plane]]
        subind_seqs = subind_seqs[:-1]
    assert isinstance(sbx_mmap, np.memmap)

    # estimate dead pixels if necessary
    if odd_row_ndead is None:
        # estimate at 3 different frames in case it changes 
        n_samps = min(len(sbx_mmap), 3)
        sample_frames = np.linspace(0, n_samps, endpoint=False, dtype=int)
        odd_row_ndead = max(_estimate_odd_row_nsaturated(sbx_mmap[frame].view(np.memmap))
                             for frame in sample_frames)
        if not quiet and odd_row_ndead == 0:
            logging.info('Found no dead pixels at left of odd rows')

    if not quiet and odd_row_ndead > 0:
        logging.info(f'Correcting {odd_row_ndead} dead pixels at left of odd rows')

    # estimate row offset if necessary
    if odd_row_offset is None:
        n_samps = min(len(sbx_mmap), 300)
        sample = sbx_mmap[np.linspace(0, n_samps, endpoint=False, dtype=int)]
        odd_row_offset = _estimate_odd_row_offset(sample)
        if not quiet and odd_row_offset == 0:
            logging.info(f'Found no line phase offset')

    if not quiet and odd_row_offset != 0:
        logging.info(f'Correcting line phase offset of {odd_row_offset}')
    
    if chunk_size is None:
        # load a contiguous block all at once
        chunk_size = n_frames_out
    chunks = [slice(start, min(start + chunk_size, n_frames_out)) for start in range(0, n_frames_out, chunk_size)]

    # create indices for loading data
    if odd_row_ndead == 0 and odd_row_offset == 0:
        # this list specifies how to index the input and output arrays when copying data.
        # format of each entry: (<tuple of spatial out-indices>, <tuple of spatial in-indices>)
        # or: (<tuple of spatial out-indices>, constant)
        # each entry is applied in order.
        inds_sets = [((), np.ix_(*subind_seqs[1:]))]
        interp_spec = None
    else:
        # ensure the selected mode is valid
        if ((out is None and not to32) or (out is not None and out.dtype.kind != 'f')) and dead_pix_mode == True:
            raise Exception('Cannot write NaN values to int array; dead_pix_mode cannot be True')
        
        if dead_pix_mode == 'min':
            # iterate over chunks to find min, note we actually call nanmax b/c values are inverted
            max_val = np.uint16(0)
            for chunk_slice in chunks:
                max_val = max(max_val, np.nanmax(sbx_mmap[subind_seqs[0][chunk_slice]]))
            dead_pix_mode = np.invert(max_val)

        # load even and odd rows separately to implement shift and correct dead pixels
        inds_sets, interp_spec = _make_inds_sets_with_corrections(n_y, n_x, subind_seqs, save_shape,
                                                                  odd_row_ndead, odd_row_offset, dead_pix_mode, interp)

    if out is None:
        out_arr = np.empty(save_shape, dtype=(np.float32 if to32 else np.uint16))
    else:
        out_arr = out
    del out  # ensure out_arr replaces out from here

    # prepare for parallel processing
    if dview is not None and len(chunks) > 1:
        if 'multiprocessing'in str(type(dview)):
            map_fn = dview.imap
        else:
            map_fn = dview.map_async
        inplace = False
    else:
        map_fn = map
        inplace = True

    args = (
        [inds_sets, sbx_mmap[subind_seqs[0][chunk_slice]], save_shape[1:], out_arr.dtype,
         out_arr[chunk_slice] if inplace else None]
        for chunk_slice in chunks
    )

    if not quiet:
        if len(chunks) > 1:
            chunks = tqdm(chunks, desc='Converting movie in chunks...', unit='chunk')
        else:
            logging.info('Converting movie...')

    for chunk, chunk_slice in zip(map_fn(_load_movie_chunk, args), chunks):
        if not inplace:
            out_arr[chunk_slice] = chunk
    if isinstance(chunks, tqdm):
        chunks.close()

    if interp and interp_spec is not None:
        _interp_offset_pixels(sbx_mmap, np.array(subind_seqs[0]), out_arr, interp_spec, dead_pix_mode, quiet=quiet)

    del sbx_mmap  # Important to close file (on Windows)

    if isinstance(out_arr, np.memmap):
        out_arr.flush()    
    return out_arr


def _sbxread_worker(args, dview=None) -> np.ndarray:
    """For calling _sbxread_helper in parallel"""
    out_mmap_args, file_in, subindices, channel, plane, to32, chunk_size, odd_row_ndead, odd_row_offset, interp, dead_pix_mode = args
    if out_mmap_args is not None:
        out = np.memmap(**out_mmap_args)
    else:
        out = None
    res = _sbxread_helper(filename=file_in, subindices=subindices, channel=channel, plane=plane, out=out, to32=to32, chunk_size=chunk_size,
                          odd_row_ndead=odd_row_ndead, odd_row_offset=odd_row_offset, interp=interp, dead_pix_mode=dead_pix_mode,
                          dview=dview, quiet=True) 
    if out is not None:
        # inplace, clean up memmap
        out.flush()
        del out
    return res


def _load_movie_chunk(args):
    inds_sets, in_arr, out_shape, out_dtype, out = args
    if out is None:
        out = np.empty((in_arr.shape[0],) + out_shape, dtype=out_dtype)
    for out_inds, in_inds in inds_sets:
        if np.isscalar(in_inds):
            chunk = in_inds
        else:
            # for advanced indexing
            #time_axis_expanded = np.expand_dims(time_axis, axis=[i+1 for i in range(len(in_inds))])

            # Note: important to copy the data here instead of making a view,
            # so the memmap can be closed (achieved by advanced indexing)
            #chunk = in_mmap[(time_axis_expanded,) + tuple(np.expand_dims(i, 0) for i in in_inds)]
            chunk = in_arr[(slice(None),) + in_inds]
            # Note: SBX files store the values strangely, it's necessary to invert each uint16 value to get the correct ones
            np.invert(chunk, out=chunk)  # avoid copying, may be large
        out[(slice(None),) + out_inds] = chunk
    return out


def _interpret_subindices(subindices: DimSubindices, dim_extent: int) -> tuple[Sequence[int], int]:
    """
    Given the extent of a dimension in the corresponding recording, obtain an iterable over subindices 
    and the step size (or 0 if the step size is not uniform).
    """
    logger = logging.getLogger("caiman")

    if isinstance(subindices, slice):
        iterable_elements = range(dim_extent)[subindices]
        skip = iterable_elements.step

        if subindices.stop is not None and np.isfinite(subindices.stop) and subindices.stop > dim_extent:
            logger.warning(f'Only {dim_extent} frames or pixels available to load ' +
                            f'(requested up to {subindices.stop})')
    else:
        iterable_elements = subindices
        if isinstance(subindices, range):
            skip = subindices.step
        else:
            skip = 0

    return iterable_elements, skip


def _get_output_shape(filename_or_shape: Union[str, tuple[int, ...]], subindices: FileSubindices
                      ) -> tuple[tuple[int, ...], tuple[Sequence[int], ...]]:
    """
    Helper to determine what shape will be loaded/saved given subindices
    Also returns back the subindices with slices transformed to ranges, for convenience
    """
    if isassignable(subindices, DimSubindices):
        dim_subindices = cast(DimSubindices, subindices)
        subindices = (dim_subindices,)
    else:
        subindices = cast(Sequence[DimSubindices], subindices)
    
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
    subinds_out: list[Sequence[int]] = []
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
    return bidiphase.item()


IndsList = tuple[np.ndarray, ...]   # (each element an output from np.ix_)
def _make_inds_sets_with_corrections(n_y: int, n_x: int, subindices: tuple[Sequence[int], ...],
                                     save_shape: tuple[int, ...], odd_row_ndead: int, odd_row_offset: int,
                                     dead_pix_mode: Union[str, bool, np.uint16], interp: bool) -> tuple[
                                         list[tuple[IndsList, Union[int, float, IndsList]]],
                                         Optional[tuple[tuple[IndsList, IndsList, IndsList], tuple[IndsList, IndsList]]]]:
    """
    Compute indices/constant values for reading and writing data given dead pixels and/or bidi offset
    Second output is the output indices that should be interpolated and extrapolated if interp is True, else None.
    """
    out_inds_y = np.arange(save_shape[1])
    out_inds_x = np.arange(save_shape[2])
    in_inds = [np.array(subind) for subind in subindices]
    in_inds_y = in_inds[1] if len(in_inds) > 1 else np.arange(n_y)
    in_inds_x = in_inds[2] if len(in_inds) > 2 else np.arange(n_x)

    b_even_row = in_inds_y % 2 == 0
    inds_sets: list[tuple[IndsList, Union[int, float, IndsList]]] = []

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
        np.ix_(out_inds_y[b_even_row], out_inds_x[e2e_mask]),
        np.ix_(in_inds_y[b_even_row], in_inds_x[e2e_mask] + even_shift, *in_inds[3:])
    ))

    o2o_mask = (odd_row_ndead <= in_inds_x + odd_shift) & (in_inds_x + odd_shift < n_x)
    inds_sets.append((
        np.ix_(out_inds_y[~b_even_row], out_inds_x[o2o_mask]),
        np.ix_(in_inds_y[~b_even_row], in_inds_x[o2o_mask] + odd_shift, *in_inds[3:])
    ))

    # wrapping pixels at ends of rows
    wrap_mask = in_inds_x + lshift >= n_x
    to_even = odd_row_offset < 0
    wrap_out_y = out_inds_y[b_even_row == to_even]
    wrap_in_y = in_inds_y[b_even_row == to_even] + (1 if to_even else -1)
    if odd_row_offset != 0:
        inds_sets.append((
            np.ix_(wrap_out_y, out_inds_x[wrap_mask]),
            np.ix_(wrap_in_y, -(in_inds_x[wrap_mask] + lshift - n_x + 1), *in_inds[3:])
        ))

    even_dead_mask = in_inds_x + even_shift < 0
    odd_dead_mask = in_inds_x + odd_shift < odd_row_ndead
    
    if interp:
        interp_mask_x = odd_dead_mask ^ even_dead_mask
        dead_mask = odd_dead_mask & even_dead_mask

        interp_even_rows = -even_shift > odd_row_ndead - odd_shift
        interp_mask_y = b_even_row if interp_even_rows else ~b_even_row

        # format: (<indices for building interpolator from input array>,
        #          <indices to assign into output array>, <indices relative to interpolator to query>)
        even_edge = -even_shift
        odd_edge = odd_row_ndead - odd_shift
        y_offset = 1 if interp_even_rows else 0
        x_offset = min(even_edge, odd_edge)
        x_range = np.arange(x_offset, max(even_edge, odd_edge))

        interp_inds = (
            np.ix_(np.arange(y_offset, n_y, 2), x_range, *in_inds[3:]),
            np.ix_(out_inds_y[interp_mask_y], out_inds_x[interp_mask_x]),
            ((in_inds_y[interp_mask_y] - y_offset)/2, in_inds_x[interp_mask_x] - x_offset,
             *[np.arange(sz) for sz in save_shape[3:]])
        )

        extrap_inds = (
            np.ix_(out_inds_y, out_inds_x[dead_mask]),
            np.ix_(in_inds_y, np.array([x_offset]), *in_inds[3:])
        )
        interp_spec = (interp_inds, extrap_inds)
    else:
        # specify how to fill all dead pixels without interpolating
        interp_spec = None

        if isinstance(dead_pix_mode, bool):
            dead_pix_mode_ext = np.nan if dead_pix_mode else 0
        else:
            dead_pix_mode_ext = dead_pix_mode    
        del dead_pix_mode

        if dead_pix_mode_ext == 'copy':
            inds_sets.append((
                np.ix_(out_inds_y[b_even_row], out_inds_x[even_dead_mask]),
                np.ix_(in_inds_y[b_even_row], np.array([0]), *in_inds[3:])
            ))

            inds_sets.append((
                np.ix_(out_inds_y[~b_even_row], out_inds_x[odd_dead_mask]),
                np.ix_(out_inds_y[~b_even_row], np.array([odd_row_ndead]), *in_inds[3:])
            ))          
        elif np.isreal(dead_pix_mode_ext):
            inds_sets.append((
                np.ix_(out_inds_y[b_even_row], out_inds_x[even_dead_mask]),
                float(dead_pix_mode_ext)
            ))

            inds_sets.append((
                np.ix_(out_inds_y[~b_even_row], out_inds_x[odd_dead_mask]),
                float(dead_pix_mode_ext)
            ))
        else:
            raise ValueError('Unrecognized dead pixel mode')
    
    return inds_sets, interp_spec
    

def _interp_offset_pixels(sbx_mmap: np.memmap, in_inds_t: np.ndarray, out: np.ndarray, 
                          interp_spec: tuple[tuple[IndsList, IndsList, IndsList], tuple[IndsList, IndsList]],
                          extrap_mode: Union[str, bool, np.uint16], quiet=False) -> None:
    """
    linearly interpolate pixels from input file (sbx_mmap[in_inds_t]) into given indices (interp_spec) of output file (out),
    taking odd_row_ndead and odd_row_offset into account. extrap_mode can be True, False, 'copy', or a float to fill with.
    """
    interp_inds, extrap_inds = interp_spec
    construct_inds, assign_inds, query_inds = interp_inds
    extrap_inds_out, extrap_inds_in = extrap_inds
    
    query_inds_grid = np.stack(np.meshgrid(*query_inds, indexing='ij'))

    if extrap_mode == False:
        mode = 'constant'
        cval = 0
    elif extrap_mode == True:
        mode = 'constant'
        cval = np.nan
    elif extrap_mode == 'copy':
        mode = 'nearest'
        cval = 0
    elif np.isreal(extrap_mode):
        mode = 'constant'
        cval = float(extrap_mode)
    else:
        raise ValueError(f'Unrecognized extrap_mode "{extrap_mode}"')

    inds_iterator = enumerate(np.squeeze(in_inds_t))
    if not quiet:
        inds_iterator = tqdm(inds_iterator, total=len(in_inds_t), desc='Doing interp/extrapolation...', unit='frame')
    orig_inds = None
    for t_out, t_in in inds_iterator:
        frame_inv = np.invert(sbx_mmap[t_in]).astype(out.dtype)
        out[t_out][assign_inds] = ndimage.map_coordinates(frame_inv[construct_inds], query_inds_grid, output=out.dtype,
                                                          order=1, mode=mode, cval=cval)
        if mode == 'constant':
            out[t_out][extrap_inds_out] = cval
        else:
            # apply interpolation result to input as well so that extrapolation is correct
            if orig_inds is None:
                # interpolate query indices against construct indices to move back to original image space
                orig_inds = tuple(
                    interpolate.interp1d(range(dim_cind.size), dim_cind, kind='linear', axis=ax,
                                        fill_value='extrapolate')(dim_qind).astype(int)  # type: ignore
                    for ax, (dim_cind, dim_qind) in enumerate(zip(construct_inds, query_inds))
                )
            frame_inv[orig_inds] = out[t_out][assign_inds]
            out[t_out][extrap_inds_out] = frame_inv[extrap_inds_in]
    if isinstance(inds_iterator, tqdm):
        inds_iterator.close()
    