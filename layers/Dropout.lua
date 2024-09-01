local Util = require "modules.Util"
local Layer = Util.Layer

-- Dropout Layer
local DropoutLayer = setmetatable({type="Dropout"}, {__index = Layer})
DropoutLayer.__index = DropoutLayer

function DropoutLayer:new(dropout_rate)
    local self = setmetatable(Layer:new(), DropoutLayer)
    self.dropout_rate = dropout_rate
    self.mask = {}
    return self
end

function DropoutLayer:get_params()
    return self
end

function DropoutLayer.set_params(params)
local self = setmetatable(Layer:new(), DropoutLayer)

for name, param in pairs(params) do
    self[name] = param
end

return self
end

function DropoutLayer:forward(input)
    self.input = input
    self.mask = {}

    local function dropout_recursive(input)
        if type(input) == "table" then
            local output = {}
            local mask = {}
            for i, v in ipairs(input) do
                output[i], mask[i] = dropout_recursive(v)
            end
            return output, mask
        else
            if math.random() > self.dropout_rate then
                return input / (1 - self.dropout_rate), 1
            else
                return 0, 0
            end
        end
    end

    self.output, self.mask = dropout_recursive(input)
    return self.output
end

function DropoutLayer:backward(gradient, learning_rate)
    local function backward_recursive(gradient, mask)
        if type(gradient) == "table" then
            local input_gradient = {}
            for i, v in ipairs(gradient) do
                input_gradient[i] = backward_recursive(v, mask[i])
            end
            return input_gradient
        else
            return gradient * mask / (1 - self.dropout_rate)
        end
    end

    return backward_recursive(gradient, self.mask)
end

return DropoutLayer