local Util = require "modules.Util"
local Layer = Util.Layer

-- Softmax Layer
local SoftmaxLayer = setmetatable({type="Softmax"}, {__index = Layer})
SoftmaxLayer.__index = SoftmaxLayer

function SoftmaxLayer:new()
    return setmetatable(Layer:new(), SoftmaxLayer)
end

function SoftmaxLayer.set_params(params)
    return SoftmaxLayer:new()
end

function SoftmaxLayer:forward(input)
    self.input = input
    self.output = Util.softmax(input)
    return self.output
end

function SoftmaxLayer:backward(gradient, learning_rate)
    -- For softmax layer, we assume the gradient is already the derivative of the loss
    -- with respect to the softmax output, so we just pass it through
    return gradient
end

return SoftmaxLayer