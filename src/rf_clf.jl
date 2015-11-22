#!/usr/bin/env julia
################################################################################
#
#  Copyright (c) 2015 Wojciech Migda
#  All rights reserved
#  Distributed under the terms of the MIT license
#
################################################################################
#
#  Filename: rf_clf.jl
#
#  Decription:
#       Classify data using RandomForest.
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

using HDF5
using DecisionTree

function pp(M)
    const MEAN = M - mean(M)
    return MEAN ./ sqrt(sumabs2(MEAN, 2))
end

function main(DATA_PATH::String)

    const X_TRAIN = pp(h5read("$(DATA_PATH)/trainX.h5", "gsv/train/X/data"))
    const Y_TRAIN = h5read("$(DATA_PATH)/trainY.h5", "gsv/train/Y/data")
    
    println("Training the classifier")
    # Train random forest with
    # 20 for number of features chosen at each random split,
    # 50 for number of trees,
    # and 1.0 for ratio of subsampling.
    model = build_forest(Y_TRAIN, X_TRAIN, 20, 50, 1.0)

    const X_TEST = pp(h5read("$(DATA_PATH)/testX.h5", "gsv/test/X/data"))
    const Y_TEST = char(apply_forest(model, X_TEST))
    println("Prediction done.")

    const TEST_LABELS = h5read("$(DATA_PATH)/testX.h5", "gsv/test/X/labels")

    # Add header option to writedlm(), https://github.com/JuliaLang/julia/issues/10284
    fid = open("$(DATA_PATH)/mySubmission.csv", "w")
    write(fid, "ID,Class\n")
    writecsv(fid, zip(TEST_LABELS, Y_TEST))
    close(fid)

    const ACCURACY = nfoldCV_forest(Y_TRAIN, X_TRAIN, 20, 50, 4, 1.0);
    println("4 fold accuracy: $(mean(ACCURACY))")
end

if ~isinteractive()
    main("/repo/kaggle/street-view-getting-started-with-julia/KAGGLE-first-steps-with-julia/data")
end
