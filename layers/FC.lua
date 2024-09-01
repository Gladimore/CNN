local Util = require "modules.Util"
local Layer = Util.Layer

-- Fully Connected Layer
local FCLayer = setmetatable({type="FC"}, {__index = Layer})
FCLayer.__index = FCLayer

function FCLayer:new(input_size, output_size)
    local self = setmetatable(Layer:new(), FCLayer)
    self.input_size = input_size
    self.output_size = output_size

    local range = math.sqrt(6 / (input_size + output_size))

    self.weights =
        Util.create_matrix(
        input_size,
        output_size,
        function()
            return Util.random(-range, range)
        end
    )
    self.biases = {}
    for i = 1, output_size do
        self.biases[i] = 0
    end

    return self
end

function FCLayer:get_params()
    return self
end

function FCLayer.set_params(params)
    local self = setmetatable(Layer:new(), FCLayer)
    
    for name, param in pairs(params) do
        self[name] = param
    end

    return self
end

function FCLayer:forward(input)
    self.input = input
    local output = {}

    --[[
    print(string.format("FC Depth: %d", Util.getTableDepth(input))) -- should be 1D
    print(string.format("FC Input: %d, Output: %d", self.input_size, self.output_size))]]
    --print("Input Length:", #input)

    for i = 1, self.output_size do
        local sum = 0
        for j = 1, self.input_size do
            sum = sum + input[j] * self.weights[j][i]
        end
        output[i] = sum + self.biases[i]
    end

    --print("FC", Util.read(output))
    self.output = output
    return output
end

function FCLayer:backward(gradient, learning_rate)
    local input_gradient = {}

    for i = 1, self.input_size do
        local sum = 0
        for j = 1, self.output_size do
            sum = sum + gradient[j] * self.weights[i][j]
            self.weights[i][j] = self.weights[i][j] - learning_rate * gradient[j] * self.input[i]
        end
        input_gradient[i] = sum
    end

    for i = 1, self.output_size do
        self.biases[i] = self.biases[i] - learning_rate * gradient[i]
    end

    return input_gradient
end

return FCLayer