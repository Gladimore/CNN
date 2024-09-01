local Util = require "modules.Util"
local Layer = Util.Layer

local floor, huge = math.floor, math.huge

-- Max Pooling Layer
local MaxPoolLayer = setmetatable({type="MaxPool"}, {__index = Layer})
MaxPoolLayer.__index = MaxPoolLayer

function MaxPoolLayer:new(pool_size, stride)
    local self = setmetatable(Layer:new(), MaxPoolLayer)
    self.pool_size = pool_size
    self.stride = stride
    return self
end

function MaxPoolLayer:get_params()
    return self
end

function MaxPoolLayer.set_params(params)
    local self = setmetatable(Layer:new(), MaxPoolLayer)

    for name, param in pairs(params) do
        self[name] = param
    end

    return self
end

function MaxPoolLayer:forward(input)
    self.input = input
    local input_depth, input_height, input_width = #input, #input[1], #input[1][1]
    local output_height = floor((input_height - self.pool_size) / self.stride) + 1
    local output_width = floor((input_width - self.pool_size) / self.stride) + 1

    local output = {}
    self.max_indices = {}

    for d = 1, input_depth do
        output[d] = Util.create_matrix(output_height, output_width)
        self.max_indices[d] = Util.create_matrix(output_height, output_width)

        for i = 1, output_height do
            for j = 1, output_width do
                local max_val = -huge
                local max_i, max_j

                for pi = 1, self.pool_size do
                    for pj = 1, self.pool_size do
                        local ii = (i - 1) * self.stride + pi
                        local jj = (j - 1) * self.stride + pj

                        if ii <= input_height and jj <= input_width then
                            local val = input[d][ii][jj]
                            if val > max_val then
                                max_val = val
                                max_i, max_j = ii, jj
                            end
                        end
                    end
                end

                output[d][i][j] = max_val
                self.max_indices[d][i][j] = {max_i, max_j}
            end
        end
    end

    self.output = output
    return output
end

function MaxPoolLayer:backward(gradient, learning_rate)
    local input_depth, input_height, input_width = #self.input, #self.input[1], #self.input[1][1]
    local output_depth, output_height, output_width = #gradient, #gradient[1], #gradient[1][1]

    local input_gradient = {}
    for d = 1, input_depth do
        input_gradient[d] = Util.create_matrix(input_height, input_width)
    end

    for d = 1, output_depth do
        for i = 1, output_height do
            for j = 1, output_width do
                local max_i, max_j = table.unpack(self.max_indices[d][i][j])
                input_gradient[d][max_i][max_j] = gradient[d][i][j]
            end
        end
    end

    --print("Max", Util.getTableDepth(input_gradient))
    return input_gradient
end

return MaxPoolLayer