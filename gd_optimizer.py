class gradient_descent_optimizer():
    def __init__(self, a=0.001, b1=0.9, b2=0.999, e=10**-8):
        self.alpha=a
        self.beta1=b1
        self.beta2=b2
        self.epsilon=e
    def adapt_lr(self, lr):
        return lr
    def adapt_gc(self, gc):
        return gc 

class Momentum(gradient_descent_optimizer):
    def __init__(self, b=0.9):
        super().__init__(self, b1=b)

    def adapt_gc(self, m_prev, g_curr):
        # w' = w - aM
        # M = Bmt-1+ (1-B)dL/dw
        return self.beta1*m_prev + (1-self.beta1)*g_curr


class Adagrad(gradient_descent_optimizer):
    def __init__(self, ee=10**-8):
        super().__init__(e=ee)

    def adapt_lr(self, lr, v_prev, g_curr):
        # w' = w - a / sqrt(v+e) * dL/dt
        # v = v t-1 + dL/dt ^ 2
        v = v_prev + g_curr ** 2
        return lr / (v + self.e)


class Adadelta(gradient_descent_optimizer): 
    pass

class Adam(gradient_descent_optimizer):
    pass