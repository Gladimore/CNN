local Util = require "modules.Util"
local Layer = Util.Layer

local floor = math.floor

-- Average Pooling Layer
local AvgPoolLayer = setmetatable({type="AvgPool"}, {__index = Layer})
AvgPoolLayer.__index = AvgPoolLayer

function AvgPoolLayer:new(pool_size, stride)
    local self = setmetatable(Layer:new(), AvgPoolLayer)
    self.pool_size = pool_size
    self.stride = stride
    return self
end

function AvgPoolLayer:get_params()
    return self
end

function AvgPoolLayer.set_params(params)
local self = setmetatable(Layer:new(), AvgPoolLayer)

for name, param in pairs(params) do
    self[name] = param
end

return self
end

function AvgPoolLayer:forward(input)
    self.input = input
    local input_depth, input_height, input_width = #input, #input[1], #input[1][1]
    local output_height = floor((input_height - self.pool_size) / self.stride) + 1
    local output_width = floor((input_width - self.pool_size) / self.stride) + 1

    local output = {}

    for d = 1, input_depth do
        output[d] = Util.create_matrix(output_height, output_width)

        for i = 1, output_height do
            for j = 1, output_width do
                local sum = 0
                local count = 0

                for pi = 1, self.pool_size do
                    for pj = 1, self.pool_size do
                        local ii = (i - 1) * self.stride + pi
                        local jj = (j - 1) * self.stride + pj

                        if ii <= input_height and jj <= input_width then
                            sum = sum + input[d][ii][jj]
                            count = count + 1
                        end
                    end
                end

                output[d][i][j] = sum / count
            end
        end
    end

    --print("Avg", Util.getTableDepth(output))

    self.output = output
    return output
end

function AvgPoolLayer:backward(gradient, learning_rate)
    local input_depth, input_height, input_width = #self.input, #self.input[1], #self.input[1][1]
    local output_depth, output_height, output_width = #gradient, #gradient[1], #gradient[1][1]

    local input_gradient = {}
    for d = 1, input_depth do
        input_gradient[d] = Util.create_matrix(input_height, input_width)
    end

    for d = 1, output_depth do
        for i = 1, output_height do
            for j = 1, output_width do
                local grad = gradient[d][i][j]
                local average_grad = grad / (self.pool_size * self.pool_size)

                for pi = 1, self.pool_size do
                    for pj = 1, self.pool_size do
                        local ii = (i - 1) * self.stride + pi
                        local jj = (j - 1) * self.stride + pj

                        if ii <= input_height and jj <= input_width then
                            input_gradient[d][ii][jj] = input_gradient[d][ii][jj] + average_grad
                        end
                    end
                end
            end
        end
    end

    return input_gradient
end

return AvgPoolLayer