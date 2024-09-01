local Util = require "modules.Util"
local Layer = Util.Layer

-- ReLU Activation Layer
local ReLULayer = setmetatable({type="ReLU"}, {__index = Layer})
ReLULayer.__index = ReLULayer

function ReLULayer:new()
    return setmetatable(Layer:new(), ReLULayer)
end

function ReLULayer.set_params(params)
    return ReLULayer:new()
end

function ReLULayer:forward(input)
    self.input = input
    local function relu_recursive(input)
        if type(input) == "table" then
            local output = {}
            for i, v in ipairs(input) do
                output[i] = relu_recursive(v)
            end
            return output
        else
            return Util.relu(input)
        end
    end
    
    self.output = relu_recursive(input)
    return self.output
end

function ReLULayer:backward(gradient, learning_rate)
    local function backward_recursive(gradient, input)
        if type(gradient) == "table" then
            local input_gradient = {}
            for i, v in ipairs(gradient) do
                input_gradient[i] = backward_recursive(v, input[i])
            end
            return input_gradient
        else
            return gradient * Util.relu_derivative(input)
        end
    end

    return backward_recursive(gradient, self.input)
end

return ReLULayer