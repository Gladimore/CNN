local ConvLayer = require "layers.Conv"
local ReLULayer = require "layers.ReLU"
local MaxPoolLayer = require "layers.MaxPool"
local FlattenLayer = require "layers.Flatten"
local FCLayer = require "layers.FC"
local DropoutLayer = require "layers.Dropout"
local AvgPoolLayer = require "layers.AvgPool"
local SoftmaxLayer = require "layers.Softmax"

local AdamOptimizer = require "Optomizers.Adam"

local Util = require "modules.Util"

local CNN = require "modules.CNN"
local time, clock = os.time, os.clock

local max = function(...)
    local max_v = -math.huge
    local max_i = -1
    local values = {...}

    for i, v in ipairs(values) do
        if v > max_v then
            max_v = v
            max_i = i
        end
    end

    return max_v, max_i
end

local function randomseed(seed)
    math.randomseed(seed or (time() + clock()))
end
randomseed()

local function createTarget(size, label)
    local t = Util.createTensor(size)
    t[label] = 1

    return t
end

function vertical_flip(data)
    -- Get the dimensions of the input data
    local dims = #data
    if dims == 0 then
        return data -- Empty data, return as is
    end

    -- Function to recursively flip the data
    local function flip_recursive(data)
        if type(data[1]) == "table" then
            -- Flip each subtable recursively
            for i = 1, #data do
                data[i] = flip_recursive(data[i])
            end
            -- Flip the outer table
            for i = 1, math.floor(#data / 2) do
                data[i], data[#data - i + 1] = data[#data - i + 1], data[i]
            end
        end
        return data
    end

    return flip_recursive(data)
end

-- Create CNN
local filepath = "data/model2.txt"
local cnn = CNN.load(filepath)

-- LeNet-5 Architecture V.0 94% accuracy :)
--[[cnn:add_layer(ConvLayer:new(1, 6, 5, 1, 2))  -- C1: Convolutional Layer (32x32x1 -> 28x28x6)
cnn:add_layer(MaxPoolLayer:new(2, 2))         -- S2: Subsampling Layer (28x28x6 -> 14x14x6)
cnn:add_layer(ReLULayer:new())

cnn:add_layer(ConvLayer:new(6, 16, 5, 1, 0)) -- C3: Convolutional Layer (14x14x6 -> 10x10x16)
cnn:add_layer(MaxPoolLayer:new(2, 2))         -- S4: Subsampling Layer (10x10x16 -> 5x5x16)
cnn:add_layer(ReLULayer:new())

cnn:add_layer(FlattenLayer:new())             -- Flatten (5x5x16 -> 400)

cnn:add_layer(FCLayer:new(400, 120))          -- Fully Connected Layer (400 -> 120)
cnn:add_layer(ReLULayer:new())                -- Activation Layer (ReLU)

cnn:add_layer(FCLayer:new(120, 84))           -- Fully Connected Layer (120 -> 84)
cnn:add_layer(ReLULayer:new())                -- Activation Layer (ReLU)

cnn:add_layer(FCLayer:new(84, 10))            -- Fully Connected Layer (84 -> 10)
cnn:add_layer(SoftmaxLayer:new())             -- Softmax Layer (10 -> 10)]]
--2.0
-- LeNet-5 Architecture in Lua

cnn:add_layer(ConvLayer:new(1, 6, 5, 1, 0)) -- C1: Convolutional Layer (28x28x1 -> 24x24x6)
cnn:add_layer(AvgPoolLayer:new(2, 2)) -- S2: Subsampling Layer (24x24x6 -> 12x12x6)
cnn:add_layer(ReLULayer:new())

cnn:add_layer(ConvLayer:new(6, 16, 5, 1, 0)) -- C3: Convolutional Layer (12x12x6 -> 8x8x16)
cnn:add_layer(AvgPoolLayer:new(2, 2)) -- S4: Subsampling Layer (8x8x16 -> 4x4x16)
cnn:add_layer(ReLULayer:new())

cnn:add_layer(ConvLayer:new(16, 120, 4, 1, 0)) -- C5: Convolutional Layer (4x4x16 -> 1x1x120)
cnn:add_layer(ReLULayer:new())
cnn:add_layer(DropoutLayer:new(0.5)) -- Dropout Layer

cnn:add_layer(FlattenLayer:new()) -- Flatten (1x1x120 -> 120)

cnn:add_layer(FCLayer:new(120, 84)) -- F6: Fully Connected Layer (120 -> 84)
cnn:add_layer(ReLULayer:new())
cnn:add_layer(DropoutLayer:new(0.5)) -- Dropout Layer

cnn:add_layer(FCLayer:new(84, 10)) -- Output Layer (84 -> 10)
cnn:add_layer(SoftmaxLayer:new())

--[[cnn:add_layer(FlattenLayer":new())
cnn:add_layer(FCLayer:new(10, 128))
cnn:add_layer(FCLayer:new(128, 10))

cnn:add_layer(SoftmaxLayer:new())]]
local df_learning = 0.0001
cnn:compile(df_learning, AdamOptimizer:new(df_learning))

-- mnist data
local mnist = require "modules.MNIST_LOADER"
local testing_images, testing_labels = mnist.TESTING()
local training_images, training_labels = mnist.TRAINING()

-- Training parameters
local epochs = #training_images * 2 -- iterate over them twice

local function padMatrix(matrix, targetRows, targetCols, padValue)
    local rows = #matrix
    local cols = #matrix[1] or 0

    -- Calculate the amount of padding needed
    local padTop = math.floor((targetRows - rows) / 2)
    local padBottom = targetRows - rows - padTop
    local padLeft = math.floor((targetCols - cols) / 2)
    local padRight = targetCols - cols - padLeft

    -- Create a new matrix with the target size
    local paddedMatrix = {}
    for r = 1, targetRows do
        paddedMatrix[r] = {}
        for c = 1, targetCols do
            -- Determine if the current position should be filled with original data or pad value
            if r > padTop and r <= padTop + rows and c > padLeft and c <= padLeft + cols then
                paddedMatrix[r][c] = matrix[r - padTop][c - padLeft]
            else
                paddedMatrix[r][c] = padValue
            end
        end
    end

    return matrix
end

local function epoch_train(epochs, training_data, training_labels)
    -- Training loop
    local start = clock()
    local total_loss = 0

    for i = 1, epochs do
        local indx = math.random(1, #training_data)
        local input = training_data[indx]
        local target = createTarget(10, training_labels[indx] + 1)

        local loss = cnn:train({padMatrix(input, 32, 32, 0)}, target)
        total_loss = total_loss + loss

        if i % 20 == 0 then
            os.execute("clear")
        end

        if i % 200 == 0 then
            cnn:save(filepath)
        end

        print(string.format("Epoch %d/%d, Loss: %f", i, epochs, loss))
    end
    local end_time = clock()
    local calculation_time = end_time - start

    print("Calculation Time:", calculation_time)
    print("Average Loss:", total_loss / epochs)
    cnn:save(filepath)
end

local function getAccuracy(rounds, training_data, training_labels)
    local correct = 0
    local divider = math.floor(rounds / 10)

    for i = 1, rounds do
        local indx = math.random(1, #training_data)

        local input = training_data[indx]
        local target = createTarget(10, training_labels[indx] + 1)

        if cnn:evaluate({padMatrix(input, 32, 32, 0)}, target) then
            correct = correct + 1
        end

        if i % divider == 0 then
            print(string.format("(%d/%d)", i, rounds))
        end
    end

    local accuracy = (correct / rounds) * 100
    print(string.format("Accuracy: %d%% (%d/%d)", accuracy, correct, rounds))
end

local function test()
    local indx = math.random(1, #images)

    local input = images[indx]
    local target = createTarget(10, labels[indx] + 1)

    local output = cnn:forward({padMatrix(input, 32, 32, 0)})

    local _, predicted = max(table.unpack(output))

    local _, actual = max(table.unpack(target))

    print("Input:", Util.read(input))
    print("Actual Label:", actual)

    print("Predicted:", predicted)
    print("Raw Output", Util.read(output))
end

epoch_train(epochs, training_images, training_labels)
--getAccuracy(1000, testing_images, testing_labels)
--test()