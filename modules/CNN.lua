-- CNN class
local CNN = {}
CNN.__index = CNN

local Util = require "modules.Util"
local Dkjson = require "modules.dkjson"

local max = function(...)
    local max_v = -math.huge
    local max_i = -1
    local values = { ... }

    for i, v in ipairs(values) do
        if v > max_v then
            max_v = v
            max_i = i
        end
    end

    return max_v, max_i
end

function CNN.new()
    local self = setmetatable({}, CNN)
    self.layers = {}
    self.optimizer = nil  -- Initialize optimizer
    self.default_learning_rate = nil  -- Initialize default learning rate
    self._loaded = false
    return self
end

function CNN:add_layer(layer, position)
    if not self._loaded or self.override then
        if position then
            table.insert(self.layers, position, layer)
        else
            table.insert(self.layers, layer)
        end
    end
end

function CNN:compile(default_learning_rate,optimizer)
    self.optimizer = optimizer
    self.default_learning_rate = default_learning_rate
end

function CNN:forward(input)
    local output = input
    for i, layer in ipairs(self.layers) do
        output = layer:forward(output)
    end
    return output
end

function CNN:backward(gradient)
    local learning_rate = self.default_learning_rate

    if self.optimizer then
        for i = #self.layers, 1, -1 do
            gradient = self.layers[i]:backward(gradient, self.optimizer:update(i, gradient))
        end
    else
        for i = #self.layers, 1, -1 do
            gradient = self.layers[i]:backward(gradient, learning_rate)
        end
    end
end

function CNN:train(input, target)
    local output = self:forward(input)
    local loss = Util.cross_entropy_loss(output, target)
    local gradient = Util.cross_entropy_loss_derivative(output, target)
    self:backward(gradient)
    return loss
end

function CNN:evaluate(input, target)
    local output = self:forward(input)
    local _, predicted = max(table.unpack(output))
    local _, actual = max(table.unpack(target))
    return predicted == actual
end

function CNN:save(filepath)
    local file = io.open(filepath, "w")
    if file then
        local serialized_layers = {}
        for i, layer in ipairs(self.layers) do
            table.insert(serialized_layers, {
                type = layer.type,
                params = layer:get_params()
            })
        end
        local json = Dkjson.encode(serialized_layers)
        file:write(json)
        file:close()
        print("Model saved successfully")
    else
        error("Unable to open file for writing: " .. filepath)
    end
end

function CNN.load(filepath, override)
    local file = io.open(filepath, "r")
    if file then
        local data = file:read("*a")
        file:close()
        local serialized_layers = Dkjson.decode(data)
        if serialized_layers then
            local self = CNN.new()
            self._loaded = true
            self._override = override or false
            for _, layer_data in ipairs(serialized_layers) do
                local layer_type = layer_data.type
                local layer = require("layers." .. layer_type).set_params(layer_data.params)
                
                print(layer.type)
                        table.insert(self.layers, layer)
            end
            print("Model loaded successfully")
            --print(Util.read(self.layers[1]))
            return self
        else
            error("Unable to decode JSON data from file: " .. filepath)
        end
    else
        error("Unable to open file for reading: " .. filepath)
    end
end

return CNN