local Util = require "modules.Util"
local Layer = Util.Layer

-- Flatten Layer
local FlattenLayer = setmetatable({type="Flatten"}, {__index = Layer})
FlattenLayer.__index = FlattenLayer

function FlattenLayer:new()
    return setmetatable(Layer:new(), FlattenLayer)
end

function FlattenLayer.set_params()
    return FlattenLayer:new()
end

function FlattenLayer:forward(input)
    self.input = input
    local output = {}

    local function flatten(x)
        if type(x) == "table" then
            for _, v in pairs(x) do
                flatten(v)
            end
        else
            table.insert(output, x)
        end
    end

    flatten(input)
    return output
end

function FlattenLayer:backward(gradient, learning_rate)
local function recursive_assign(input, gradient, index)
    if type(input) ~= "table" then
        return gradient[index], index + 1
    end

    local output = {}
    for i = 1, #input do
        output[i], index = recursive_assign(input[i], gradient, index)
    end
    return output, index
end

local input_gradient, _ = recursive_assign(self.input, gradient, 1)
return input_gradient
end

return FlattenLayer