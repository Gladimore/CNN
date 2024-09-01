local Util = {}

local floor, exp, max, log, min = math.floor, math.exp, math.max, math.log, math.min

-- Utility functions
function Util.random(min, max)
    return min + math.random() * (max - min)
end

function Util.create_matrix(rows, cols, init_value)
    local matrix = {}
    for i = 1, rows do
        matrix[i] = {}
        for j = 1, cols do
            matrix[i][j] = type(init_value) == "function" and init_value() or init_value or 0
        end
    end
    return matrix
end

function Util.matrix_multiply(a, b)
    local rows_a, cols_a = #a, #a[1]
    local rows_b, cols_b = #b, #b[1]

    if cols_a ~= rows_b then
        error("Cannot multiply matrices: dimensions do not match")
    end

    local result = Util.create_matrix(rows_a, cols_b)

    for i = 1, rows_a do
        for j = 1, cols_b do
            local sum = 0
            for k = 1, cols_a do
                sum = sum + a[i][k] * b[k][j]
            end
            result[i][j] = sum
        end
    end

    return result
end

function Util.convolve2d(input, kernel, stride, padding)
    local input_height, input_width = #input, #input[1]
    local kernel_size = #kernel
    local output_height = floor((input_height - kernel_size + 2 * padding) / stride) + 1
    local output_width = floor((input_width - kernel_size + 2 * padding) / stride) + 1

    local output = Util.create_matrix(output_height, output_width)

    for i = 1, output_height do
        for j = 1, output_width do
            local sum = 0
            for ki = 1, kernel_size do
                for kj = 1, kernel_size do
                    local ii = (i - 1) * stride + ki - padding
                    local jj = (j - 1) * stride + kj - padding
                    if ii > 0 and ii <= input_height and jj > 0 and jj <= input_width then
                        sum = sum + input[ii][jj] * kernel[ki][kj]
                    end
                end
            end
            output[i][j] = sum
        end
    end

    return output
end

-- Activation functions
function Util.relu(x)
    return max(0, x)
end

function Util.relu_derivative(x)
    return x > 0 and 1 or 0
end

function Util.softmax(x)
    local exp_values = {}
    local sum_exp = 0
    for i = 1, #x do
        exp_values[i] = exp(x[i])
        sum_exp = sum_exp + exp_values[i]
    end

    local softmax_values = {}
    for i = 1, #x do
        softmax_values[i] = exp_values[i] / sum_exp
    end

    return softmax_values
end

-- Loss functions
function Util.cross_entropy_loss(predictions, targets)
    local loss = 0
    for i = 1, #predictions do
        loss = loss - targets[i] * log(predictions[i])
    end
    return loss
end

function Util.cross_entropy_loss_derivative(predictions, targets)
    local derivative = {}
    for i = 1, #predictions do
        derivative[i] = predictions[i] - targets[i]
    end
    return derivative
end

-- Layer base class
Util.Layer = {}
Util.Layer.__index = Util.Layer

function Util.Layer:new()
    local self = setmetatable({}, Util.Layer)
    return self
end

function Util.Layer:forward(input)
    error("Not implemented")
end

function Util.Layer:backward(gradient, learning_rate)
    error("Not implemented")
end

function Util.Layer:get_params()
    --error("Not implemented")
    return {}
end

function Util.Layer.set_params(params)
    error("Not implemented")
end

function Util.quickGet(self, ...)
    local output = {}
    local names = { ... }

    for _, name in pairs(names) do
        output[name] = self[name]
    end

    return output
end

function Util.getTableDepth(tbl)
    local function depth(tbl)
        if type(tbl) ~= "table" then
            return 0
        end

        local maxDepth = 0
        for _, v in pairs(tbl) do
            local currentDepth = depth(v)
            if currentDepth > maxDepth then
                maxDepth = currentDepth
            end
        end

        return maxDepth + 1
    end

    return depth(tbl)
end

-- Data handling functions
function Util.create_random_data(num_samples, input_shape, num_classes)
    local data = {}
    for i = 1, num_samples do
        local input = {}
        for j = 1, input_shape[1] do
            input[j] = {}
            for k = 1, input_shape[2] do
                input[j][k] = {}
                for l = 1, input_shape[3] do
                    input[j][k][l] = math.random()
                end
            end
        end

        local target = {}
        for j = 1, num_classes do
            target[j] = 0
        end
        target[math.random(1, num_classes)] = 1

        data[i] = {input, target}
    end
    return data
end

function Util.create_batches(data, batch_size)
    local batches = {}
    for i = 1, #data, batch_size do
        local batch = {}
        for j = i, min(i + batch_size - 1, #data) do
            table.insert(batch, data[j])
        end
        table.insert(batches, batch)
    end
    return batches
end

function Util.createTensor(...)
    local dimensions = {...}

    local function createSubTensor(dims)
        if #dims == 0 then
            return 0 -- or any default value you want
        end

        local subTensor = {}
        local size = table.remove(dims, 1)

        for i = 1, size do
            subTensor[i] = createSubTensor({table.unpack(dims)})
        end

        return subTensor
    end

    return createSubTensor(dimensions)
end

local insert = table.insert

function Util.read(v, spaces, usesemicolon, depth)
    if type(v) ~= "table" then
        return tostring(v)
    elseif not next(v) then
        return "{}"
    end

    spaces = spaces or 4
    depth = depth or 1

    local space = (" "):rep(depth * spaces)
    local sep = usesemicolon and ";" or ","
    local concatenationBuilder = {"{"}

    for k, x in next, v do
        insert(
            concatenationBuilder,
            ("\n%s[%s] = %s%s"):format(
                space,
                type(k) == "number" and tostring(k) or ('"%s"'):format(tostring(k)),
                Util.read(x, spaces, usesemicolon, depth + 1),
                sep
            )
        )
    end

    local s = table.concat(concatenationBuilder)
    return ("%s\n%s}"):format(s:sub(1, -2), space:sub(1, -spaces - 1))
end

function Util.zeros_like(tensor)
-- Returns a tensor of zeros with the same shape as the input tensor
if type(tensor) == "number" then
    return 0
elseif type(tensor) == "table" then
    local zeros = {}
    for i = 1, #tensor do
        zeros[i] = Util.zeros_like(tensor[i])
    end
    return zeros
else
    error("Unsupported type for zeros_like: " .. type(tensor))
end
end

function Util.add(tensor1, tensor2)
-- Element-wise addition of two tensors
if type(tensor1) == "number" and type(tensor2) == "number" then
    return tensor1 + tensor2
elseif type(tensor1) == "table" and type(tensor2) == "table" then
    local result = {}
    for i = 1, #tensor1 do
        result[i] = Util.add(tensor1[i], tensor2[i])
    end
    return result
else
    error("Mismatched types for add: " .. type(tensor1) .. " and " .. type(tensor2))
end
end

function Util.mul(scalar, tensor)
-- Scalar multiplication of a tensor
if type(tensor) == "number" then
    return scalar * tensor
elseif type(tensor) == "table" then
    local result = {}
    for i = 1, #tensor do
        result[i] = Util.mul(scalar, tensor[i])
    end
    return result
else
    error("Unsupported type for mul: " .. type(tensor))
end
end

function Util.div(tensor1, tensor2)
-- Element-wise division of two tensors
if type(tensor1) == "number" and type(tensor2) == "number" then
    return tensor1 / tensor2
elseif type(tensor1) == "table" and type(tensor2) == "table" then
    local result = {}
    for i = 1, #tensor1 do
        result[i] = Util.div(tensor1[i], tensor2[i])
    end
    return result
else
    error("Mismatched types for div: " .. type(tensor1) .. " and " .. type(tensor2))
end
end

function Util.sub(tensor1, tensor2)
-- Element-wise subtraction of two tensors
if type(tensor1) == "number" and type(tensor2) == "number" then
    return tensor1 - tensor2
elseif type(tensor1) == "table" and type(tensor2) == "table" then
    local result = {}
    for i = 1, #tensor1 do
        result[i] = Util.sub(tensor1[i], tensor2[i])
    end
    return result
else
    error("Mismatched types for sub: " .. type(tensor1) .. " and " .. type(tensor2))
end
end

function Util.sqrt(tensor)
-- Element-wise square root of a tensor
if type(tensor) == "number" then
    return math.sqrt(tensor)
elseif type(tensor) == "table" then
    local result = {}
    for i = 1, #tensor do
        result[i] = Util.sqrt(tensor[i])
    end
    return result
else
    error("Unsupported type for sqrt: " .. type(tensor))
end
end

function Util.square(tensor)
-- Element-wise square of a tensor
if type(tensor) == "number" then
    return tensor * tensor
elseif type(tensor) == "table" then
    local result = {}
    for i = 1, #tensor do
        result[i] = Util.square(tensor[i])
    end
    return result
else
    error("Unsupported type for square: " .. type(tensor))
end
end

return Util