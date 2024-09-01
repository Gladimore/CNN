local Util = require "modules.Util"
local Layer = Util.Layer

-- Adam Optimizer
local AdamOptimizer = {}
AdamOptimizer.__index = AdamOptimizer

function AdamOptimizer:new(learning_rate, beta1, beta2, epsilon)
    local self = setmetatable({}, AdamOptimizer)
    self.learning_rate = learning_rate or 0.001
    self.beta1 = beta1 or 0.9
    self.beta2 = beta2 or 0.999
    self.epsilon = epsilon or 1e-7
    self.m = {}  -- First moment estimate
    self.v = {}  -- Second moment estimate
    self.t = 0   -- Timestep
    return self
end

function AdamOptimizer:update(grad)
    self.t = self.t + 1

    -- Initialize m and v for this parameter if not already done
    if not self.m[1] then self.m[1] = Util.zeros_like(grad) end
    if not self.v[1] then self.v[1] = Util.zeros_like(grad) end

    -- Update biased first moment estimate (element-wise operation)
    self.m[1] = Util.add(Util.mul(self.beta1, self.m[1]), Util.mul(1 - self.beta1, grad))

    -- Update biased second moment estimate (element-wise operation)
    self.v[1] = Util.add(Util.mul(self.beta2, self.v[1]), Util.mul(1 - self.beta2, Util.square(grad)))

    -- Compute bias-corrected first moment estimate (element-wise operation)
    local m_hat = Util.div(self.m[1], (1 - self.beta1 ^ self.t))

    -- Compute bias-corrected second moment estimate (element-wise operation)
    local v_hat = Util.div(self.v[1], (1 - self.beta2 ^ self.t))

    -- Calculate the adjusted learning rate (element-wise operation)
    local adjusted_learning_rate = Util.div(self.learning_rate, Util.add(Util.sqrt(v_hat), self.epsilon))

    return adjusted_learning_rate
end

return AdamOptimizer