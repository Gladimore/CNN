-- Advanced CNN for Number Instance Segmentation

local unpack = table.unpack

-- Utility functions
local function create_matrix(rows, cols, init_value)
    local mat = {}
    for i = 1, rows do
        mat[i] = {}
        for j = 1, cols do
            mat[i][j] = type(init_value) == "function" and init_value() or init_value or 0
        end
    end
    return mat
end

local function create_3d_matrix(depth, rows, cols, init_value)
    local mat = {}
    for d = 1, depth do
        mat[d] = create_matrix(rows, cols, init_value)
    end
    return mat
end

local function random(min, max)
    return min + math.random() * (max - min)
end

local function relu(x)
    return math.max(0, x)
end

local function relu_derivative(x)
    return x > 0 and 1 or 0
end

local function softmax(x)
    local exp_sum = 0
    local max_x = math.max(unpack(x))
    for i = 1, #x do
        exp_sum = exp_sum + math.exp(x[i] - max_x)
    end
    local result = {}
    for i = 1, #x do
        result[i] = math.exp(x[i] - max_x) / exp_sum
    end
    return result
end

-- Layer interface
local Layer = {}
function Layer:new()
    local layer = {}
    setmetatable(layer, self)
    self.__index = self
    return layer
end

function Layer:forward(input) end
function Layer:backward(output_gradient, learning_rate) end

-- Convolutional Layer
local ConvLayer = Layer:new()
function ConvLayer:new(input_depth, input_height, input_width, output_depth, kernel_size, stride, padding)
    local layer = Layer.new(self)
    layer.input_depth = input_depth
    layer.input_height = input_height
    layer.input_width = input_width
    layer.output_depth = output_depth
    layer.kernel_size = kernel_size
    layer.stride = stride
    layer.padding = padding
    layer.output_height = math.floor((input_height + 2*padding - kernel_size) / stride) + 1
    layer.output_width = math.floor((input_width + 2*padding - kernel_size) / stride) + 1
    layer.weights = create_3d_matrix(output_depth, kernel_size, kernel_size, function() return random(-0.5, 0.5) end)
    layer.biases = create_matrix(1, output_depth, function() return random(-0.5, 0.5) end)
    return layer
end

function ConvLayer:forward(input)
    self.input = input
    self.output = create_3d_matrix(self.output_depth, self.output_height, self.output_width)
    
    for d = 1, self.output_depth do
        for i = 1, self.output_height do
            for j = 1, self.output_width do
                local sum = self.biases[1][d]
                for di = 1, self.input_depth do
                    for ki = 1, self.kernel_size do
                        for kj = 1, self.kernel_size do
                            local ii = (i - 1) * self.stride + ki - self.padding
                            local jj = (j - 1) * self.stride + kj - self.padding
                            if ii > 0 and ii <= self.input_height and jj > 0 and jj <= self.input_width then
                                sum = sum + self.weights[d][ki][kj] * input[di][ii][jj]
                            end
                        end
                    end
                end
                self.output[d][i][j] = relu(sum)
            end
        end
    end
    
    return self.output
end

function ConvLayer:backward(output_gradient, learning_rate)
    local input_gradient = create_3d_matrix(self.input_depth, self.input_height, self.input_width)
    local weight_gradient = create_3d_matrix(self.output_depth, self.kernel_size, self.kernel_size)
    local bias_gradient = create_matrix(1, self.output_depth)
    
    for d = 1, self.output_depth do
        for i = 1, self.output_height do
            for j = 1, self.output_width do
                local grad = output_gradient[d][i][j] * relu_derivative(self.output[d][i][j])
                bias_gradient[1][d] = bias_gradient[1][d] + grad
                
                for di = 1, self.input_depth do
                    for ki = 1, self.kernel_size do
                        for kj = 1, self.kernel_size do
                            local ii = (i - 1) * self.stride + ki - self.padding
                            local jj = (j - 1) * self.stride + kj - self.padding
                            if ii > 0 and ii <= self.input_height and jj > 0 and jj <= self.input_width then
                                weight_gradient[d][ki][kj] = weight_gradient[d][ki][kj] + grad * self.input[di][ii][jj]
                                input_gradient[di][ii][jj] = input_gradient[di][ii][jj] + grad * self.weights[d][ki][kj]
                            end
                        end
                    end
                end
            end
        end
    end
    
    -- Update weights and biases
    for d = 1, self.output_depth do
        for i = 1, self.kernel_size do
            for j = 1, self.kernel_size do
                self.weights[d][i][j] = self.weights[d][i][j] - learning_rate * weight_gradient[d][i][j]
            end
        end
        self.biases[1][d] = self.biases[1][d] - learning_rate * bias_gradient[1][d]
    end
    
    return input_gradient
end

-- Pooling Layer
local PoolLayer = Layer:new()
function PoolLayer:new(input_depth, input_height, input_width, pool_size, stride)
    local layer = Layer.new(self)
    layer.input_depth = input_depth
    layer.input_height = input_height
    layer.input_width = input_width
    layer.pool_size = pool_size
    layer.stride = stride
    layer.output_height = math.floor((input_height - pool_size) / stride) + 1
    layer.output_width = math.floor((input_width - pool_size) / stride) + 1
    return layer
end

function PoolLayer:forward(input)
    self.input = input
    self.output = create_3d_matrix(self.input_depth, self.output_height, self.output_width)
    self.max_indices = create_3d_matrix(self.input_depth, self.output_height, self.output_width)
    
    for d = 1, self.input_depth do
        for i = 1, self.output_height do
            for j = 1, self.output_width do
                local max_val = -math.huge
                local max_i, max_j
                for pi = 1, self.pool_size do
                    for pj = 1, self.pool_size do
                        local ii = (i - 1) * self.stride + pi
                        local jj = (j - 1) * self.stride + pj
                        if input[d][ii][jj] > max_val then
                            max_val = input[d][ii][jj]
                            max_i, max_j = ii, jj
                        end
                    end
                end
                self.output[d][i][j] = max_val
                self.max_indices[d][i][j] = {max_i, max_j}
            end
        end
    end
    
    return self.output
end

function PoolLayer:backward(output_gradient)
    local input_gradient = create_3d_matrix(self.input_depth, self.input_height, self.input_width)
    
    for d = 1, self.input_depth do
        for i = 1, self.output_height do
            for j = 1, self.output_width do
                local max_i, max_j = unpack(self.max_indices[d][i][j])
                input_gradient[d][max_i][max_j] = output_gradient[d][i][j]
            end
        end
    end
    
    return input_gradient
end

-- Flatten Layer
local FlattenLayer = Layer:new()
function FlattenLayer:new(input_depth, input_height, input_width)
    local layer = Layer.new(self)
    layer.input_depth = input_depth
    layer.input_height = input_height
    layer.input_width = input_width
    layer.output_size = input_depth * input_height * input_width
    return layer
end

function FlattenLayer:forward(input)
    self.input = input
    self.output = {}
    local index = 1
    for d = 1, self.input_depth do
        for i = 1, self.input_height do
            for j = 1, self.input_width do
                self.output[index] = input[d][i][j]
                index = index + 1
            end
        end
    end
    return self.output
end

function FlattenLayer:backward(output_gradient)
    local input_gradient = create_3d_matrix(self.input_depth, self.input_height, self.input_width)
    local index = 1
    for d = 1, self.input_depth do
        for i = 1, self.input_height do
            for j = 1, self.input_width do
                input_gradient[d][i][j] = output_gradient[index]
                index = index + 1
            end
        end
    end
    return input_gradient
end

-- Fully Connected Layer
local FCLayer = Layer:new()
function FCLayer:new(input_size, output_size)
    local layer = Layer.new(self)
    layer.input_size = input_size
    layer.output_size = output_size
    layer.weights = create_matrix(output_size, input_size, function() return random(-0.5, 0.5) end)
    layer.biases = create_matrix(output_size, 1, function() return random(-0.5, 0.5) end)
    return layer
end

function FCLayer:forward(input)
    self.input = input
    self.output = {}
    for i = 1, self.output_size do
        local sum = self.biases[i][1]
        for j = 1, self.input_size do
            sum = sum + self.weights[i][j] * input[j]
        end
        self.output[i] = sum
    end
    return self.output
end

function FCLayer:backward(output_gradient, learning_rate)
    local input_gradient = {}
    for i = 1, self.input_size do
        input_gradient[i] = 0
        for j = 1, self.output_size do
            input_gradient[i] = input_gradient[i] + output_gradient[j] * self.weights[j][i]
        end
    end
    
    for i = 1, self.output_size do
        for j = 1, self.input_size do
            self.weights[i][j] = self.weights[i][j] - learning_rate * output_gradient[i] * self.input[j]
        end
        self.biases[i][1] = self.biases[i][1] - learning_rate * output_gradient[i]
    end
    
    return input_gradient
end

-- Softmax Layer
local SoftmaxLayer = Layer:new()
function SoftmaxLayer:new()
    return Layer.new(self)
end

function SoftmaxLayer:forward(input)
    self.input = input
    self.output = softmax(input)
    return self.output
end

function SoftmaxLayer:backward(output_gradient)
    local input_gradient = {}
    for i = 1, #self.input do
        input_gradient[i] = 0
        for j = 1, #self.input do
            if i == j then
                input_gradient[i] = input_gradient[i] + output_gradient[j] * self.output[i] * (1 - self.output[i])
            else
                input_gradient[i] = input_gradient[i] - output_gradient[j] * self.output[i] * self.output[j]
            end
        end
    end
    return input_gradient
end

-- Model
local Model = {}
function Model:new()
    local model = {
        layers = {},
        loss = 0
    }
    setmetatable(model, self)
    self.__index = self
    return model
end

function Model:add_layer(layer)
    table.insert(self.layers, layer)
end

function Model:forward(input)
    local output = input
    for _, layer in ipairs(self.layers) do
        output = layer:forward(output)
    end
    return output
end

function Model:backward(target, learning_rate)
    local output_gradient = self:calculate_output_gradient(target)
    for i = #self.layers, 1, -1 do
        output_gradient = self.layers[i]:backward(output_gradient, learning_rate)
    end
end

function Model:calculate_output_gradient(target)
    local output = self.layers[#self.layers].output
    local gradient = {}
    self.loss = 0
    
    for i = 1, #output do
        gradient[i] = output[i] - target[i]
        self.loss = self.loss + 0.5 * (gradient[i] ^ 2)
    end
    
    return gradient
end

-- Instance Segmentation
local function perform_instance_segmentation(model, input)
    local output = model:forward(input)
    local segmentation = create_matrix(#input[1], #input[1][1])
    local threshold = 0.5
    
    for i = 1, #segmentation do
        for j = 1, #segmentation[1] do
            local max_value = 0
            local max_class = 0
            for k = 1, #output do
                if output[k] > max_value then
                    max_value = output[k]
                    max_class = k
                end
            end
            if max_value > threshold then
                segmentation[i][j] = max_class
            else
                segmentation[i][j] = 0  -- Background
            end
        end
    end
    
    return segmentation
end

-- Training function
local function train(model, train_data, train_labels, epochs, learning_rate)
    for epoch = 1, epochs do
        local total_loss = 0
        
        for i = 1, #train_data do
            model:forward(train_data[i])
            model:backward(train_labels[i], learning_rate)
            total_loss = total_loss + model.loss

            -- Print progress every 100 iterations
            if i % 100 == 0 then
                print(string.format("Epoch %d, Progress: %.2f%%", epoch, (i / #train_data) * 100))
                coroutine.yield()  -- Allow for periodic yields
            end
        end

        local avg_loss = total_loss / #train_data
        print(string.format("Epoch %d, Loss: %.6f", epoch, avg_loss))

        coroutine.yield()  -- Allow for periodic yields
    end
end
-- Main execution
local function main()
    -- Create model
    local model = Model:new()
    model:add_layer(ConvLayer:new(1, 28, 28, 32, 3, 1, 1))  -- Convolutional layer
    model:add_layer(PoolLayer:new(32, 28, 28, 2, 2))  -- Max pooling layer
    model:add_layer(ConvLayer:new(32, 14, 14, 64, 3, 1, 1))  -- Convolutional layer
    model:add_layer(PoolLayer:new(64, 14, 14, 2, 2))  -- Max pooling layer
    model:add_layer(FlattenLayer:new(64, 7, 7))  -- Flatten layer
    model:add_layer(FCLayer:new(64 * 7 * 7, 128))  -- Fully connected layer
    model:add_layer(FCLayer:new(128, 10))  -- Output layer (10 classes for digits 0-9)
    model:add_layer(SoftmaxLayer:new())  -- Softmax activation for output

    -- Generate dummy training data (replace with real data)
    local train_data = {}
    local train_labels = {}
    for i = 1, 1000 do
        train_data[i] = create_3d_matrix(1, 28, 28, function() return random(0, 1) end)
        local label = {}
        for j = 1, 10 do
            label[j] = 0
        end
        label[math.random(1, 10)] = 1
        train_labels[i] = label
    end

    -- Train the model
    local co = coroutine.create(function()
        train(model, train_data, train_labels, 10, 0.01)
    end)

    while coroutine.status(co) ~= "dead" do
        local success, error = coroutine.resume(co)
        if not success then
            print("Error during training:", error)
            break
        end
    end
    -- Perform instance segmentation on a test image
    local test_image = create_3d_matrix(1, 28, 28, function() return random(0, 1) end)
    local segmentation = perform_instance_segmentation(model, test_image)

    -- Print segmentation result
    print("Segmentation result:")
    for i = 1, #segmentation do
        local row = ""
        for j = 1, #segmentation[i] do
            row = row .. segmentation[i][j] .. " "
        end
        print(row)
    end
end

-- Run the main function
main()

