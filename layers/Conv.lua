local Util = require "modules.Util"
local Layer = Util.Layer

local floor = math.floor

-- Convolutional Layer
local ConvLayer = setmetatable({type="Conv"}, {__index = Layer})
ConvLayer.__index = ConvLayer

function ConvLayer:new(input_channels, output_channels, kernel_size, stride, padding)
    local self = setmetatable(Layer:new(), ConvLayer)
    self.input_channels = input_channels
    self.output_channels = output_channels
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding

    local range = math.sqrt(6/input_channels)

    self.kernels = {}
    self.biases = {}
    for i = 1, output_channels do
        self.kernels[i] = {}
        for j = 1, input_channels do
            self.kernels[i][j] =
                Util.create_matrix(
                kernel_size,
                kernel_size,
                function()
                    return Util.random(-range, range)
                end
            )
        end
        self.biases[i] = 0
    end

    return self
end

function ConvLayer:get_params()
    return Util.quickGet(self, "input_channels", "output_channels", "kernel_size", "stride", "padding", "kernels", "biases")
end

function ConvLayer.set_params(params)
local self = setmetatable(Layer:new(), ConvLayer)

for name, param in pairs(params) do
    self[name] = param
end

return self
end

function ConvLayer:forward(input)
    self.input = input
    local input_depth, input_height, input_width = #input, #input[1], #input[1][1]
    --print("Conv Data", input_depth, input_height, input_width)
    
    local output_height = floor((input_height - self.kernel_size + 2 * self.padding) / self.stride) + 1
    local output_width = floor((input_width - self.kernel_size + 2 * self.padding) / self.stride) + 1

    local output = {}
    for i = 1, self.output_channels do
        output[i] = Util.create_matrix(output_height, output_width)
        for j = 1, input_depth do
            local conv_result = Util.convolve2d(input[j], self.kernels[i][j], self.stride, self.padding)
            --print("conv result", Util.read(conv_result))

            for y = 1, output_height do
                for x = 1, output_width do
                    output[i][y][x] = (output[i][y][x] or 0) + conv_result[y][x]
                end
            end
        end
        for y = 1, output_height do
            for x = 1, output_width do
                output[i][y][x] = Util.relu(output[i][y][x] + self.biases[i])
            end
        end
    end

    --print(Util.read(output))
    self.output = output
    return output
end

function ConvLayer:backward(gradient, learning_rate)
    local input_depth, input_height, input_width = #self.input, #self.input[1], #self.input[1][1]
    local output_depth, output_height, output_width = #gradient, #gradient[1], #gradient[1][1]

    local kernel_gradient = {}
    local bias_gradient = {}
    local input_gradient = {}

    for i = 1, self.output_channels do
        kernel_gradient[i] = {}
        for j = 1, self.input_channels do
            kernel_gradient[i][j] = Util.create_matrix(self.kernel_size, self.kernel_size)
        end
        bias_gradient[i] = 0
    end

    for i = 1, input_depth do
        input_gradient[i] = Util.create_matrix(input_height, input_width)
    end

    for i = 1, output_depth do
        for j = 1, self.input_channels do
            for y = 1, output_height do
                for x = 1, output_width do
                    local grad_value = gradient[i][y][x] * Util.relu_derivative(self.output[i][y][x])
                    bias_gradient[i] = bias_gradient[i] + grad_value

                    for ky = 1, self.kernel_size do
                        for kx = 1, self.kernel_size do
                            local iy = (y - 1) * self.stride + ky - self.padding
                            local ix = (x - 1) * self.stride + kx - self.padding
                            if iy > 0 and iy <= input_height and ix > 0 and ix <= input_width then
                                kernel_gradient[i][j][ky][kx] =
                                    kernel_gradient[i][j][ky][kx] + grad_value * self.input[j][iy][ix]
                                input_gradient[j][iy][ix] =
                                    input_gradient[j][iy][ix] + grad_value * self.kernels[i][j][ky][kx]
                            end
                        end
                    end
                end
            end
        end
    end

    for i = 1, self.output_channels do
        for j = 1, self.input_channels do
            for y = 1, self.kernel_size do
                for x = 1, self.kernel_size do
                    self.kernels[i][j][y][x] = self.kernels[i][j][y][x] - learning_rate * kernel_gradient[i][j][y][x]
                end
            end
        end
        self.biases[i] = self.biases[i] - learning_rate * bias_gradient[i]
    end

    return input_gradient
end

return ConvLayer