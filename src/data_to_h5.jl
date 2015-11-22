#!/usr/bin/env julia
################################################################################
#
#  Copyright (c) 2015 Wojciech Migda
#  All rights reserved
#  Distributed under the terms of the MIT license
#
################################################################################
#
#  Filename: data_to_h5.jl
#
#  Decription:
#       Convert BMP data and put it along feature labels into H5 storages.
#
#  Authors:
#       Wojciech Migda
#
################################################################################
#
#  History:
#  --------
#  Date         Who  Ticket     Description
#  ----------   ---  ---------  ------------------------------------------------
#  2015-11-22   wm              Initial version
#
################################################################################

using Images
using ColorTypes
using HDF5

function read_bmp_data(FEAT_LABELS, IMAGE_SIZE, PATH)

    x = zeros(Float32, size(FEAT_LABELS, 1), IMAGE_SIZE)

    for (index, img_id) in enumerate(FEAT_LABELS[:, 1])
        # Read image file
        img = imread("$(PATH)/$(img_id).Bmp")

        img = convert(Array{Gray}, float32(img))
        img = convert(Array{Float32}, img)

        # Transform image matrix to a vector and store
        # it in data matrix
        x[index, :] = reshape(img, 1, IMAGE_SIZE)
        #break
    end 

    return x
end

function main(DATA_PATH::String)

    const IMAGE_SIZE = 20 * 20
    
    const TRAIN_FEAT_LABELS = readdlm("$(DATA_PATH)/trainLabels.csv", ',', ASCIIString, header=true)[1] # [2] is the header
    const X_TRAIN = read_bmp_data(TRAIN_FEAT_LABELS, IMAGE_SIZE, "$(DATA_PATH)/trainResized")
    const Y_TRAIN = map(x -> int(x[1]), TRAIN_FEAT_LABELS[:, 2])

    const TEST_FEAT_LABELS = readdlm("$(DATA_PATH)/sampleSubmission.csv", ',', ASCIIString, header=true)[1]
    const X_TEST = read_bmp_data(TEST_FEAT_LABELS, IMAGE_SIZE, "$(DATA_PATH)/testResized")
    
    h5write("$(DATA_PATH)/trainX.h5", "gsv/train/X/data", X_TRAIN)
    h5write("$(DATA_PATH)/trainY.h5", "gsv/train/Y/data", Y_TRAIN)

    h5write("$(DATA_PATH)/testX.h5", "gsv/test/X/data", X_TEST)
    fid = h5open("$(DATA_PATH)/testX.h5", "r+")
    labels_dset = d_create(fid, "gsv/test/X/labels", datatype(Int), dataspace(size(TEST_FEAT_LABELS[:, 1])))
    labels_dset[:] = int(TEST_FEAT_LABELS[:, 1])
    close(fid)

end

if ~isinteractive()
    main("/repo/kaggle/street-view-getting-started-with-julia/KAGGLE-first-steps-with-julia/data")
end
