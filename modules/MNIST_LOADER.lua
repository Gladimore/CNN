local os = require 'os'
local clock = os.clock
local open = io.open

local function readInt32(file)
    local bytes = file:read(4)
    return (bytes:byte(1) * 16777216) + (bytes:byte(2) * 65536) + (bytes:byte(3) * 256) + bytes:byte(4)
end

local function loadMNISTImages(filename)
    local startclock = clock()

    local file = open(filename, 'rb')
    if file == nil then
        error('Cannot open file ' .. filename)
    end

    local magicNumber = readInt32(file)
    if magicNumber ~= 2051 then
        error('Invalid magic number in file ' .. filename)
    end

    local numberOfImages = readInt32(file)
    local numberOfRows = readInt32(file)
    local numberOfColumns = readInt32(file)
    local totalPixels = numberOfRows * numberOfColumns
    local dataSize = numberOfImages * totalPixels

    local allImageData = file:read(dataSize)
    file:close()

    local images = {}
    local index = 1

    for _ = 1, numberOfImages do
        local image = {}
        for r = 1, numberOfRows do
            local row = {}
            for c = 1, numberOfColumns do
                local pixel = allImageData:byte(index) / 255
                row[c] = pixel
                index = index + 1
            end
            image[r] = row
        end
        images[#images + 1] = image
    end

    local endclock = clock()
    print('Time taken to load images: ' .. (endclock - startclock) .. ' seconds')

    return images, numberOfRows, numberOfColumns
end

local function loadMNISTLabels(filename)
    local startclock = clock()

    local file = open(filename, 'rb')
    if file == nil then
        error('Cannot open file ' .. filename)
    end

    local magicNumber = readInt32(file)
    if magicNumber ~= 2049 then
        error('Invalid magic number in file ' .. filename)
    end

    local numberOfLabels = readInt32(file)
    local labelData = file:read(numberOfLabels)
    file:close()

    local labels = {}
    for i = 1, numberOfLabels do
        labels[i] = labelData:byte(i)
    end

    local endclock = clock()
    print('Time taken to load labels: ' .. (endclock - startclock) .. ' seconds')

    return labels
end

--[[local function makeFile(name, data)
    local file = io.open(name, "w")
    file:write(data)
    file:close()
end

-- For rblx
local c = require "modules/Compressor"
local d = require "modules/dkjson"

makeFile("images.lua", c.compress(d.encode({unpack(images, 1, 1000)})))
makeFile("labels.lua", d.encode(labels))
]]

local Mnist = {}
Mnist.__index = Mnist

local data_path = "data/mnist/"

local paths = {
    training = {
        images = "train/train-images.idx3-ubyte",
        labels = "train/train-labels.idx1-ubyte"
    },
    testing = {
        images = "test/t10k-images.idx3-ubyte",
        labels = "test/t10k-labels.idx1-ubyte"
    }
}

local format = "Loaded %d Images and %d Labels, of Size %d x %d"

local function c(configure_path, path)
    return (configure_path or data_path) .. path
end

function Mnist._load(image_path, label_path, configure_path)
    local images, rows, cols = loadMNISTImages(c(configure_path, image_path))
    local labels = loadMNISTLabels(c(configure_path, label_path))

    print(string.format(format, #images, #labels, rows, cols))
    return images, labels
end

function Mnist.TRAINING(img_path, label_path, configure_path)
    img_path = img_path or paths.training.images
    label_path = label_path or paths.training.labels
    return Mnist._load(img_path, label_path, configure_path)
end

function Mnist.TESTING(img_path, label_path, configure_path)
    img_path = img_path or paths.testing.images
    label_path = label_path or paths.testing.labels
    return Mnist._load(img_path, label_path, configure_path)
end

return Mnist