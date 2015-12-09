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
#  2015-12-09   wm              Redesign. +CLI, +image scaling, +parallel
#
################################################################################

addprocs(div(CPU_CORES, 2))

@everywhere using Images
@everywhere using ColorTypes
@everywhere using HDF5
using ArgParse


function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        #"--opt1"
        #    help = "an option with an argument"
        "--data-dir", "-D"
            help = "directory with input CSV files, BMP 'train' and 'test' subfolders, and where H5 will be stored"
            arg_type = String
            required = true
        "--do-x-labels"
            help = "store X features file labels"
            arg_type = Bool
            default = false
        "--do-y"
            help = "store 'y' train vector"
            arg_type = Bool
            default = false
        "--image-size", "-s"
            help = "size to which BMP files will be resized (in pixels). 0 or negative value to skip generating feature vectors"
            arg_type = Int
            default = 0

        #"arg1"
        #    help = "a positional argument"
        #    required = true
    end

    return parse_args(s)
end


@everywhere function image_producer(DATA_PATH, FILE_NAMES)

    function core()
        for img_name in FILE_NAMES
            produce(imread("$(DATA_PATH)/$(img_name).Bmp"))
        end
    end

    Task(() -> core())

end


@everywhere function image_resizer(IMAGES, SHAPE)

    function core()
        for img in IMAGES
            produce(Images.imresize(img, SHAPE))
        end
    end
    
    Task(() -> core())

end


@everywhere function image_converter(IMAGES, VLEN)
    
    function core()
        for img in IMAGES
            img |>
                img -> convert(Array{Gray}, float32(img)) |>
                img -> convert(Array{Float32}, img) |>
                img -> reshape(img, 1, VLEN) |>
                produce
        end
    end
    
    Task(() -> core())
    
end


function transform(
    DATA_PATH::String,
    do_x_labels::Bool,
    do_y::Bool,
    IMSZ::Int
    )

    const TRAIN_FEAT_LABELS = readdlm(DATA_PATH * "/trainLabels.csv", ',', ASCIIString, header=true)[1] # [2] is the header
    const TEST_FEAT_LABELS = readdlm(DATA_PATH * "/sampleSubmission.csv", ',', ASCIIString, header=true)[1]
    const train_y = map(x -> int(x[1]), TRAIN_FEAT_LABELS[:, 2])

    if do_y
        h5write("$(DATA_PATH)/trainY.h5", "gsv/train/Y/data", train_y)
    end

    if do_x_labels
        h5write("$(DATA_PATH)/testXlabels.h5", "gsv/test/X/labels", TEST_FEAT_LABELS[:, 1])
    end

    if IMSZ > 0
        const work_descr = [
            {
                "subfolder" => "train",
                "files_labels" => TRAIN_FEAT_LABELS[:, 1],
                "h5prefix" => "train",
                "dataset" => "gsv/train/X/data"
            }
            {
                "subfolder" => "test",
                "files_labels" => TEST_FEAT_LABELS[:, 1],
                "h5prefix" => "test",
                "dataset" => "gsv/test/X/data"}
        ]
    
        @sync @parallel for work = work_descr
            const X = image_producer(DATA_PATH * "/" * work["subfolder"], work["files_labels"]) |>
                i -> image_resizer(i, (IMSZ, IMSZ)) |>
                i -> image_converter(i, IMSZ ^ 2) |>
                i -> vcat(i...)

            h5write(DATA_PATH * "/" * work["h5prefix"] * "X_$(IMSZ).h5", work["dataset"], X)
        end
    end

end


function main()
    parsed_args = parse_commandline()
    println("Parsed args:")
    for (arg,val) in parsed_args
        println("  $arg  =>  $val")
    end
    
    transform(
        parsed_args["data-dir"],
        parsed_args["do-x-labels"],
        parsed_args["do-y"],
        parsed_args["image-size"]
    )
end


if ~isinteractive()
    main()
end
